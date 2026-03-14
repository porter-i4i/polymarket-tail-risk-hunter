"""
Polymarket Tail Risk Hunter — Operational Dashboard

Production-grade Streamlit dashboard that reads from the bot's SQLite database.
Run: streamlit run dashboard.py
Read-only with respect to trading actions. Degrades gracefully on empty DB.
"""
import os
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.portfolio_tracker import Database
from modules.types import BotPhase

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Polymarket Tail Risk Hunter",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Compact metrics */
    [data-testid="stMetric"] {
        background: rgba(28, 131, 225, 0.04);
        border: 1px solid rgba(28, 131, 225, 0.1);
        border-radius: 8px;
        padding: 12px 16px 8px 16px;
    }
    [data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        color: #888 !important;
    }
    /* Table density */
    .stDataFrame td, .stDataFrame th {
        font-size: 0.82rem !important;
        padding: 4px 8px !important;
    }
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.95rem;
    }
    /* Phase badge colors */
    .phase-calibration { color: #1e88e5; font-weight: 600; }
    .phase-scaled { color: #43a047; font-weight: 600; }
    .phase-full { color: #8e24aa; font-weight: 600; }
    .phase-circuit { color: #e65100; font-weight: 600; }
    .phase-killed { color: #c62828; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── DB connection ────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "data/polybot.db")
REFRESH_SECONDS = 30


@st.cache_resource
def get_db() -> Database | None:
    """Open a read-only Database handle. Returns None if file doesn't exist."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        return Database(DB_PATH)
    except Exception:
        return None


def _safe(fn, default=None):
    """Call fn, returning default on any exception — prevents dashboard crash."""
    try:
        result = fn()
        return result if result is not None else default
    except Exception:
        return default


# ─── Phase helpers ────────────────────────────────────────────────────────────
PHASE_NAMES = {
    1: "Calibration",
    2: "Scaled",
    3: "Full",
    4: "Circuit Breaker",
    5: "Killed",
}

PHASE_ICONS = {
    1: "🔵",
    2: "🟢",
    3: "🟣",
    4: "🟠",
    5: "🔴",
}


def phase_label(phase: int) -> str:
    icon = PHASE_ICONS.get(phase, "⚪")
    name = PHASE_NAMES.get(phase, f"Unknown ({phase})")
    return f"{icon} Phase {phase}: {name}"


def mode_label(db: Database) -> str:
    """Determine dry-run vs live from recent orders or env."""
    orders = _safe(lambda: db.get_recent_orders(1), [])
    if orders and orders[0].get("is_dry_run"):
        return "🧪 Dry Run"
    if os.environ.get("DRY_RUN", "true").lower() in ("true", "1", "yes"):
        return "🧪 Dry Run"
    return "🔴 LIVE"


# ─── Empty state helper ──────────────────────────────────────────────────────
def empty_state(msg: str = "No data available yet. Run the bot to populate."):
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    db = get_db()

    # Sidebar
    st.sidebar.title("Tail Risk Hunter")
    st.sidebar.caption("Operational Dashboard")

    if db is None:
        st.sidebar.warning("Database not found")
        st.error(
            f"Database file not found at `{DB_PATH}`. "
            "Initialize it with: `python -c \"from modules.portfolio_tracker import Database; Database()\"`"
        )
        return

    # Auto-refresh
    refresh = st.sidebar.selectbox(
        "Auto-refresh", [30, 60, 120, 300], index=0, format_func=lambda x: f"{x}s"
    )
    st.sidebar.divider()

    # Navigation
    page = st.sidebar.radio(
        "View",
        [
            "Status & KPIs",
            "Positions",
            "Predictions",
            "Cycle & Operations",
            "Calibration & Performance",
            "Alerts & Errors",
        ],
        index=0,
    )

    # Last updated timestamp
    st.sidebar.divider()
    st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    # Auto-refresh via rerun fragment
    st_autorefresh(refresh)

    # Route to page
    if page == "Status & KPIs":
        render_status_kpis(db)
    elif page == "Positions":
        render_positions(db)
    elif page == "Predictions":
        render_predictions(db)
    elif page == "Cycle & Operations":
        render_operations(db)
    elif page == "Calibration & Performance":
        render_calibration(db)
    elif page == "Alerts & Errors":
        render_alerts(db)


def st_autorefresh(interval_sec: int):
    """Inject JS-based auto-refresh since st_autorefresh may not be installed."""
    ms = interval_sec * 1000
    st.markdown(
        f"""
        <script>
            setTimeout(function(){{
                window.parent.postMessage({{type: 'streamlit:rerun'}}, '*');
            }}, {ms});
        </script>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  STATUS & KPIs
# ═══════════════════════════════════════════════════════════════════════════════
def render_status_kpis(db: Database):
    st.header("Status & KPIs")

    # Top row: mode, phase, balance, heartbeat
    today = date.today()
    phase = _safe(lambda: db.get_current_phase(), 1)
    balance = _safe(lambda: db.get_current_balance(), 0.0)
    mode = mode_label(db)

    col_mode, col_phase, col_balance, col_peak = st.columns(4)
    col_mode.metric("Mode", mode)
    col_phase.metric("Phase", phase_label(phase))
    col_balance.metric("Balance", f"${balance:,.2f}")
    peak = _safe(lambda: db.get_peak_balance(), balance)
    col_peak.metric("Peak Balance", f"${peak:,.2f}")

    st.divider()

    # KPI row
    open_count = _safe(lambda: db.get_open_position_count(), 0)
    daily_spend = _safe(lambda: db.get_daily_spend(today), 0.0)
    daily_bets = _safe(lambda: db.get_daily_bet_count(today), 0)
    realized_pnl = _safe(lambda: db.get_total_realized_pnl(), 0.0)
    unrealized_pnl = _safe(lambda: db.get_total_unrealized_pnl(), 0.0)
    rolling7 = _safe(lambda: db.get_rolling_pnl(7), 0.0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Open Positions", open_count)
    c2.metric("Daily Spend", f"${daily_spend:,.2f}")
    c3.metric("Daily Bets", daily_bets)
    c4.metric("Realized PnL", f"${realized_pnl:+,.2f}")
    c5.metric("Unrealized PnL", f"${unrealized_pnl:+,.2f}")
    c6.metric("7d Rolling PnL", f"${rolling7:+,.2f}")

    st.divider()

    # Calibration + Sharpe quick view
    stats_rows = _safe(lambda: db.get_daily_stats_history(1), [])
    if stats_rows:
        latest = stats_rows[0]
        cal_score = latest.get("calibration_score")
        sharpe = latest.get("sharpe_30d")
        consec_loss = _safe(lambda: db.get_consecutive_loss_days(), 0)

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(
            "Calibration Score",
            f"{cal_score:.3f}" if cal_score is not None else "N/A",
        )
        cc2.metric(
            "Sharpe (30d)",
            f"{sharpe:.3f}" if sharpe is not None else "N/A",
        )
        cc3.metric("Consecutive Loss Days", consec_loss)
    else:
        st.info("No daily stats recorded yet.")

    # Phase transitions
    transitions = _safe(lambda: db.get_phase_transitions_history(5), [])
    if transitions:
        st.subheader("Recent Phase Transitions")
        for t in transitions:
            fp = t.get("from_phase", "?")
            tp = t.get("to_phase", "?")
            reason = t.get("trigger_reason", "")
            ts = t.get("timestamp", "")
            st.markdown(
                f"**{PHASE_NAMES.get(fp, fp)} → {PHASE_NAMES.get(tp, tp)}** "
                f"— {reason}  \n<small>{ts}</small>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════
def render_positions(db: Database):
    st.header("Positions")

    tab_open, tab_closed = st.tabs(["Open Positions", "Closed Positions"])

    with tab_open:
        positions = _safe(lambda: db.get_open_positions_list(), [])
        if not positions:
            empty_state("No open positions.")
        else:
            df = pd.DataFrame(positions)
            display_cols = [
                c for c in [
                    "id", "market_question", "side", "avg_price", "shares",
                    "current_price", "unrealized_pnl", "opened_at", "end_date",
                ] if c in df.columns
            ]
            if display_cols:
                df_display = df[display_cols].copy()
                # Calculate age
                if "opened_at" in df_display.columns:
                    try:
                        df_display["age_hours"] = (
                            (pd.Timestamp.utcnow() - pd.to_datetime(df_display["opened_at"], errors="coerce"))
                            .dt.total_seconds() / 3600
                        ).round(1)
                    except Exception:
                        pass
                # Truncate question
                if "market_question" in df_display.columns:
                    df_display["market_question"] = df_display["market_question"].str[:80]
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                st.caption(f"{len(positions)} open position(s)")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_closed:
        closed = _safe(lambda: db.get_closed_positions(50), [])
        if not closed:
            empty_state("No closed positions yet.")
        else:
            df = pd.DataFrame(closed)
            display_cols = [
                c for c in [
                    "id", "market_question", "side", "avg_price", "shares",
                    "unrealized_pnl", "exit_reason", "opened_at", "closed_at",
                ] if c in df.columns
            ]
            if display_cols:
                df_display = df[display_cols].copy()
                if "market_question" in df_display.columns:
                    df_display["market_question"] = df_display["market_question"].str[:80]
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                st.caption(f"{len(closed)} closed position(s)")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTIONS / MODEL
# ═══════════════════════════════════════════════════════════════════════════════
def render_predictions(db: Database):
    st.header("Predictions & Model")

    preds = _safe(lambda: db.get_recent_predictions(100), [])
    if not preds:
        empty_state("No predictions recorded yet.")
        return

    df = pd.DataFrame(preds)

    # Summary metrics
    total_preds = len(df)
    resolved = df[df["outcome"].notna()]
    unresolved = df[df["outcome"].isna()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions (recent)", total_preds)
    c2.metric("Resolved", len(resolved))
    c3.metric("Unresolved", len(unresolved))

    st.divider()

    # Predictions table
    st.subheader("Recent Predictions")
    display_cols = [
        c for c in [
            "id", "condition_id", "ai_probability_raw", "ai_probability_debiased",
            "ai_probability_adjusted", "market_price", "ai_confidence",
            "base_rate", "phase", "outcome", "scored_at",
        ] if c in df.columns
    ]
    if display_cols:
        df_display = df[display_cols].copy()
        # Shorten condition_id
        if "condition_id" in df_display.columns:
            df_display["condition_id"] = df_display["condition_id"].str[:12] + "…"
        # Format probabilities
        for col in ["ai_probability_raw", "ai_probability_debiased",
                     "ai_probability_adjusted", "market_price", "base_rate"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                )
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Scatter: AI adjusted vs market price
    if "ai_probability_adjusted" in df.columns and "market_price" in df.columns:
        st.subheader("AI Adjusted vs Market Price")
        df_plot = df.dropna(subset=["ai_probability_adjusted", "market_price"]).copy()
        if len(df_plot) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot["market_price"],
                y=df_plot["ai_probability_adjusted"],
                mode="markers",
                marker=dict(size=6, opacity=0.7, color="#1e88e5"),
                name="Predictions",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="#888"),
                name="Perfect calibration",
            ))
            fig.update_layout(
                xaxis_title="Market Price",
                yaxis_title="AI Adjusted Probability",
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("Not enough data for scatter plot.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CYCLE & OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def render_operations(db: Database):
    st.header("Cycle & Operations")

    orders = _safe(lambda: db.get_recent_orders(100), [])
    if not orders:
        empty_state("No orders recorded yet. Run the bot to see cycle data.")
        return

    df = pd.DataFrame(orders)

    # Status breakdown
    st.subheader("Order Status Breakdown")
    if "status" in df.columns:
        status_counts = df["status"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pending", int(status_counts.get("pending", 0)))
        c2.metric("Filled", int(status_counts.get("filled", 0)))
        c3.metric("Cancelled", int(status_counts.get("cancelled", 0)))
        c4.metric("Resolved", int(status_counts.get("resolved", 0)))
    st.divider()

    # Dry run indicator
    if "is_dry_run" in df.columns:
        dry_count = int(df["is_dry_run"].sum())
        live_count = len(df) - dry_count
        st.caption(f"Orders shown: {dry_count} dry-run, {live_count} live")

    # Orders table
    st.subheader("Recent Orders")
    display_cols = [
        c for c in [
            "order_id", "market_question", "side", "price", "size",
            "total_cost", "status", "ai_score_adjusted", "ai_confidence",
            "phase", "is_dry_run", "created_at",
        ] if c in df.columns
    ]
    if display_cols:
        df_display = df[display_cols].copy()
        if "market_question" in df_display.columns:
            df_display["market_question"] = df_display["market_question"].str[:60]
        if "order_id" in df_display.columns:
            df_display["order_id"] = df_display["order_id"].str[:16]
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Daily orders chart
    if "created_at" in df.columns:
        st.subheader("Orders per Day")
        try:
            df["date"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date
            daily_orders = df.groupby("date").size().reset_index(name="count")
            if len(daily_orders) > 0:
                fig = go.Figure(go.Bar(
                    x=daily_orders["date"],
                    y=daily_orders["count"],
                    marker_color="#1e88e5",
                ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Orders",
                    height=300,
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION & PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
def render_calibration(db: Database):
    st.header("Calibration & Performance")

    stats = _safe(lambda: db.get_daily_stats_history(90), [])
    if not stats:
        empty_state("No daily stats yet. The bot records these after each cycle.")
        return

    df = pd.DataFrame(stats)
    # Reverse so oldest first for charts
    df = df.iloc[::-1].reset_index(drop=True)

    # Summary from latest
    latest = stats[0]  # most recent (original order)
    st.subheader("Latest Daily Stats")
    lc1, lc2, lc3, lc4, lc5 = st.columns(5)
    lc1.metric("Date", latest.get("date", "—"))
    lc2.metric("Bets", latest.get("total_bets", 0))
    lc3.metric("Spent", f"${latest.get('total_spent', 0):,.2f}")
    lc4.metric("Net PnL", f"${latest.get('net_pnl', 0):+,.2f}")
    win_rate = latest.get("win_rate", 0)
    lc5.metric("Win Rate", f"{win_rate:.1%}" if win_rate else "—")

    st.divider()

    # Calibration score trend
    if "calibration_score" in df.columns and df["calibration_score"].notna().any():
        st.subheader("Calibration Score Trend")
        df_cal = df.dropna(subset=["calibration_score"])
        if len(df_cal) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_cal["date"],
                y=df_cal["calibration_score"],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color="#43a047", width=2),
                name="Calibration Score",
            ))
            # Phase 2/3 thresholds
            fig.add_hline(y=0.60, line_dash="dash", line_color="#1e88e5",
                          annotation_text="Phase 2 threshold")
            fig.add_hline(y=0.65, line_dash="dash", line_color="#8e24aa",
                          annotation_text="Phase 3 threshold")
            fig.add_hline(y=0, line_dash="dot", line_color="#c62828",
                          annotation_text="Kill zone")
            fig.update_layout(
                yaxis_title="Brier Skill Score",
                height=350,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Net PnL trend
    if "net_pnl" in df.columns:
        st.subheader("Daily Net PnL")
        fig = go.Figure()
        colors = ["#43a047" if v >= 0 else "#c62828" for v in df["net_pnl"]]
        fig.add_trace(go.Bar(
            x=df["date"],
            y=df["net_pnl"],
            marker_color=colors,
            name="Net PnL",
        ))
        fig.update_layout(
            yaxis_title="USD",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cumulative PnL
    if "net_pnl" in df.columns:
        st.subheader("Cumulative PnL")
        df["cum_pnl"] = df["net_pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["cum_pnl"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#1e88e5", width=2),
            name="Cumulative PnL",
        ))
        fig.update_layout(
            yaxis_title="USD",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Balance over time
    if "balance" in df.columns:
        st.subheader("Balance History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["balance"],
            mode="lines",
            line=dict(color="#8e24aa", width=2),
            name="Balance",
        ))
        fig.update_layout(
            yaxis_title="USD",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sharpe trend
    if "sharpe_30d" in df.columns and df["sharpe_30d"].notna().any():
        st.subheader("30-Day Sharpe Ratio")
        df_sharpe = df.dropna(subset=["sharpe_30d"])
        if len(df_sharpe) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_sharpe["date"],
                y=df_sharpe["sharpe_30d"],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color="#ff9800", width=2),
                name="Sharpe (30d)",
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="#888")
            fig.update_layout(
                yaxis_title="Sharpe Ratio",
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERTS & ERRORS
# ═══════════════════════════════════════════════════════════════════════════════
def render_alerts(db: Database):
    st.header("Alerts & Errors")

    alerts: list[dict] = []

    # 1. Kill-condition proximity warnings
    balance = _safe(lambda: db.get_current_balance(), 0.0)
    peak = _safe(lambda: db.get_peak_balance(), balance)
    bankroll = float(os.environ.get("TOTAL_BANKROLL", "1000"))

    if peak > 0:
        drawdown = (peak - balance) / peak
        if drawdown >= 0.30:
            alerts.append({
                "severity": "CRITICAL" if drawdown >= 0.35 else "WARNING",
                "category": "Drawdown",
                "message": f"Drawdown from peak: {drawdown:.1%} (kill at 40%)",
            })

    total_loss = bankroll - balance
    if total_loss >= 400:
        alerts.append({
            "severity": "CRITICAL" if total_loss >= 450 else "WARNING",
            "category": "Absolute Loss",
            "message": f"Total loss: ${total_loss:,.0f} (kill at $500)",
        })

    consec = _safe(lambda: db.get_consecutive_loss_days(), 0)
    if consec >= 10:
        alerts.append({
            "severity": "CRITICAL" if consec >= 12 else "WARNING",
            "category": "Loss Streak",
            "message": f"Consecutive loss days: {consec} (kill at 14)",
        })

    # 2. Phase transitions as informational alerts
    transitions = _safe(lambda: db.get_phase_transitions_history(5), [])
    for t in transitions:
        tp = t.get("to_phase", 0)
        severity = "CRITICAL" if tp in (4, 5) else "INFO"
        alerts.append({
            "severity": severity,
            "category": "Phase Transition",
            "message": (
                f"{PHASE_NAMES.get(t.get('from_phase', 0), '?')} → "
                f"{PHASE_NAMES.get(tp, '?')}: {t.get('trigger_reason', '')} "
                f"({t.get('timestamp', '')})"
            ),
        })

    # 3. Failed / cancelled orders
    orders = _safe(lambda: db.get_recent_orders(50), [])
    cancelled = [o for o in orders if o.get("status") == "cancelled"]
    if cancelled:
        alerts.append({
            "severity": "WARNING",
            "category": "Orders",
            "message": f"{len(cancelled)} cancelled orders in recent history",
        })

    # 4. Stale positions (open > 7 days without end_date change)
    positions = _safe(lambda: db.get_open_positions_list(), [])
    now = datetime.utcnow()
    stale_positions = []
    for p in positions:
        opened = p.get("opened_at", "")
        if opened:
            try:
                opened_dt = datetime.fromisoformat(opened.replace("Z", "+00:00").replace("+00:00", ""))
                age_days = (now - opened_dt).days
                if age_days > 7:
                    stale_positions.append(p)
            except Exception:
                pass
    if stale_positions:
        alerts.append({
            "severity": "INFO",
            "category": "Stale Positions",
            "message": f"{len(stale_positions)} position(s) open for >7 days",
        })

    # 5. Empty state
    if not alerts:
        st.success("No alerts or warnings. System appears healthy.")
        return

    # Render alerts grouped by severity
    for sev in ["CRITICAL", "WARNING", "INFO"]:
        sev_alerts = [a for a in alerts if a["severity"] == sev]
        if not sev_alerts:
            continue
        for a in sev_alerts:
            if sev == "CRITICAL":
                st.error(f"**{a['category']}**: {a['message']}")
            elif sev == "WARNING":
                st.warning(f"**{a['category']}**: {a['message']}")
            else:
                st.info(f"**{a['category']}**: {a['message']}")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
else:
    # Streamlit runs the file as a script, not via __main__
    main()
