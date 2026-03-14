#!/usr/bin/env python3
"""
Pre-flight health check for Polymarket Tail Risk Hunter.

Verifies config, dependencies, database, and module imports before starting
the bot or transitioning to live mode.

Usage:
    python scripts/health_check.py             # standard checks
    python scripts/health_check.py --live      # include live-mode checks
    python scripts/health_check.py --strict    # exit 1 on any warning
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

# Allow running from repo root or from scripts/ directory
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

PASS = "\033[32mPASS\033[0m"
WARN = "\033[33mWARN\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results: list[tuple[str, str, str]] = []   # (check_name, status, detail)


def check(name: str, fn) -> bool:
    """Run a check function. Returns True if passed."""
    try:
        detail = fn()
        if detail is None:
            detail = ""
        results.append((name, PASS, detail))
        return True
    except AssertionError as exc:
        results.append((name, FAIL, str(exc)))
        return False
    except Exception as exc:
        results.append((name, FAIL, f"{type(exc).__name__}: {exc}"))
        return False


def warn(name: str, fn) -> bool:
    """Run a check — WARN instead of FAIL on exception."""
    try:
        detail = fn()
        if detail is None:
            detail = ""
        results.append((name, PASS, detail))
        return True
    except Exception as exc:
        results.append((name, WARN, f"{type(exc).__name__}: {exc}"))
        return True  # warnings don't fail the run


# ---------------------------------------------------------------------------
# CHECK FUNCTIONS
# ---------------------------------------------------------------------------

def _check_python_version() -> str:
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (3, 11), f"Python {major}.{minor} < 3.11"
    return f"Python {major}.{minor}"


def _check_repo_root() -> str:
    for fname in ("main_loop.py", "config.py", "requirements.txt"):
        p = Path(repo_root) / fname
        assert p.exists(), f"Missing file: {p}"
    return repo_root


def _check_env_file() -> str:
    p = Path(repo_root) / ".env"
    if p.exists():
        return ".env present"
    return ".env not found (will use environment variables)"


def _check_dotenv_loads() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(repo_root) / ".env")
    except ImportError:
        pass  # python-dotenv optional
    return "OK"


def _check_required_imports() -> str:
    modules = [
        "httpx", "tenacity", "sklearn", "loguru",
        "numpy", "dotenv",
    ]
    missing = []
    for m in modules:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    assert not missing, f"Missing packages: {missing}. Run: pip install -r requirements.txt"
    return f"All {len(modules)} core packages importable"


def _check_bot_modules() -> str:
    mods = [
        "config",
        "modules.types",
        "modules.portfolio_tracker",
        "modules.rate_limiter",
        "modules.market_scanner",
        "modules.ai_scoring",
        "modules.scoring_cache",
        "modules.risk_manager",
        "modules.order_executor",
        "modules.exit_manager",
        "modules.calibration_tracker",
        "modules.calibration_adjuster",
        "modules.tail_event_debiaser",
        "modules.adverse_selection_detector",
        "modules.order_book_health",
        "modules.fee_calculator",
        "modules.signal_aggregator",
        "modules.correlation_manager",
        "modules.notifications",
        "modules.news_engine",
        "modules.position_monitor",
    ]
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as exc:
            failed.append(f"{m}: {exc}")
    assert not failed, "Import failures:\n  " + "\n  ".join(failed)
    return f"{len(mods)} bot modules importable"


def _check_config_loads() -> str:
    from config import load_config
    cfg = load_config()
    assert isinstance(cfg, dict), "load_config() did not return dict"
    assert "DRY_RUN" in cfg, "DRY_RUN missing from config"
    return f"dry_run={cfg['DRY_RUN']}, bankroll={cfg.get('TOTAL_BANKROLL')}"


def _check_config_valid() -> str:
    from config import load_config, validate_config
    cfg = load_config()
    errors = validate_config(cfg)
    # In dry-run mode, missing POLYMARKET_PRIVATE_KEY is expected; filter it
    if cfg.get("DRY_RUN", True):
        errors = [e for e in errors if "POLYMARKET_PRIVATE_KEY" not in e]
    assert not errors, "Config errors:\n  " + "\n  ".join(errors)
    return "Config valid"


def _check_database() -> str:
    from config import load_config
    cfg = load_config()
    db_path = cfg.get("DB_PATH", "data/polybot.db")
    # Initialise (idempotent)
    from modules.portfolio_tracker import Database
    db = Database(db_path=db_path)
    phase = db.get_current_phase()
    balance = db.get_current_balance()
    open_pos = db.get_open_position_count()
    return f"phase={phase}, balance=${balance:.2f}, open_positions={open_pos}"


def _check_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    assert key, "ANTHROPIC_API_KEY not set (required for AI scoring)"
    assert len(key) > 20, "ANTHROPIC_API_KEY looks too short"
    return f"Key present (length={len(key)})"


def _check_telegram_config() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    assert token, "TELEGRAM_BOT_TOKEN not set"
    assert chat, "TELEGRAM_CHAT_ID not set"
    assert ":" in token, "TELEGRAM_BOT_TOKEN format looks wrong (expected 'id:hash')"
    return "Token and chat ID present"


def _check_private_key_set() -> str:
    """Live-only check: private key must be present."""
    key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    assert key, "POLYMARKET_PRIVATE_KEY not set (required for live mode)"
    assert len(key) >= 60, "POLYMARKET_PRIVATE_KEY seems too short (check format)"
    return f"Key present (length={len(key)})"


def _check_dry_run_false() -> str:
    """Live-only check: DRY_RUN must be false."""
    from config import load_config
    cfg = load_config()
    assert not cfg.get("DRY_RUN", True), "DRY_RUN=true — set to false for live mode"
    return "DRY_RUN=false"


def _check_data_dir() -> str:
    data_dir = Path(repo_root) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    assert data_dir.is_dir(), f"Cannot create data directory: {data_dir}"
    return str(data_dir)


def _check_logs_dir() -> str:
    logs_dir = Path(repo_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    assert logs_dir.is_dir(), f"Cannot create logs directory: {logs_dir}"
    return str(logs_dir)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Polymarket bot pre-flight health check")
    parser.add_argument("--live", action="store_true", help="Include live-mode checks (private key, DRY_RUN=false)")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any warning")
    args = parser.parse_args()

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(repo_root) / ".env")
    except ImportError:
        pass

    print("=" * 60)
    print("  Polymarket Tail Risk Hunter — Health Check")
    print("=" * 60)

    # Core checks
    check("Python version", _check_python_version)
    check("Repo structure", _check_repo_root)
    warn("Env file", _check_env_file)
    warn("dotenv load", _check_dotenv_loads)
    check("Core packages", _check_required_imports)
    check("Bot module imports", _check_bot_modules)
    check("Config loads", _check_config_loads)
    check("Config valid", _check_config_valid)
    check("Data directory", _check_data_dir)
    check("Logs directory", _check_logs_dir)
    check("Database initialise", _check_database)
    check("ANTHROPIC_API_KEY", _check_anthropic_key)
    check("Telegram config", _check_telegram_config)

    # Live-mode specific checks
    if args.live:
        print("\n  [Live mode checks]")
        check("POLYMARKET_PRIVATE_KEY", _check_private_key_set)
        check("DRY_RUN=false", _check_dry_run_false)

    # Print results table
    print("\n  Results:")
    print("  " + "-" * 56)
    failed = 0
    warned = 0
    for name, status, detail in results:
        suffix = f"  — {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if "FAIL" in status:
            failed += 1
        if "WARN" in status:
            warned += 1

    print("  " + "-" * 56)
    total = len(results)
    passed = total - failed - warned
    print(f"  {passed}/{total} passed, {warned} warnings, {failed} failures")
    print("=" * 60)

    if failed > 0:
        print("\n  ACTION REQUIRED: Fix failures above before starting the bot.")
        return 1
    if warned > 0 and args.strict:
        print("\n  STRICT MODE: Warnings treated as failures.")
        return 1
    if failed == 0 and warned == 0:
        print("\n  All checks passed. Ready to run.")
    else:
        print("\n  Passed with warnings. Review warnings before going live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
