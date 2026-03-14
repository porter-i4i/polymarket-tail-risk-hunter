"""Tests for modules/correlation_manager.py — TF-IDF clustering and exposure lookup."""

import pytest

from modules.correlation_manager import CorrelationManager


def _pos(cid: str, question: str) -> dict:
    return {"condition_id": cid, "market_question": question}


# === UPDATE CLUSTERS ===


class TestUpdateClusters:
    def test_empty_positions(self):
        mgr = CorrelationManager()
        mgr.update_clusters([])
        assert mgr.clusters == {}

    def test_single_position(self):
        mgr = CorrelationManager()
        mgr.update_clusters([_pos("a", "Will Bitcoin hit 100k?")])
        assert mgr.clusters == {0: ["a"]}

    def test_two_similar_questions_same_cluster(self):
        mgr = CorrelationManager()
        positions = [
            _pos("a", "Will Biden resign from presidency?"),
            _pos("b", "Will Biden leave the presidency early?"),
        ]
        mgr.update_clusters(positions)
        # Should be clustered together (high TF-IDF similarity)
        all_cids = []
        for cluster in mgr.clusters.values():
            all_cids.extend(cluster)
        assert set(all_cids) == {"a", "b"}
        # Both in same cluster
        assert len(mgr.clusters) == 1

    def test_two_dissimilar_questions_different_clusters(self):
        mgr = CorrelationManager()
        positions = [
            _pos("a", "Will Bitcoin cryptocurrency hit one hundred thousand dollars?"),
            _pos("b", "Will France win the soccer World Cup tournament championship?"),
        ]
        mgr.update_clusters(positions)
        # Should be in different clusters (very different topics)
        assert len(mgr.clusters) >= 2

    def test_all_assigned(self):
        mgr = CorrelationManager()
        positions = [
            _pos("a", "Will it rain tomorrow in NYC?"),
            _pos("b", "Will Apple stock rise 10%?"),
            _pos("c", "Will rain fall in New York City tomorrow?"),
        ]
        mgr.update_clusters(positions)
        all_cids = []
        for cluster in mgr.clusters.values():
            all_cids.extend(cluster)
        assert set(all_cids) == {"a", "b", "c"}

    def test_rebuild_replaces_previous(self):
        mgr = CorrelationManager()
        mgr.update_clusters([_pos("a", "Will X happen?")])
        assert "a" in mgr.clusters[0]
        mgr.update_clusters([_pos("b", "Will Y happen?")])
        all_cids = []
        for cluster in mgr.clusters.values():
            all_cids.extend(cluster)
        assert "a" not in all_cids
        assert "b" in all_cids


# === GET CLUSTER EXPOSURE ===


class _FakeDB:
    """Minimal stub for db.get_position_cost()."""

    def __init__(self, costs: dict[str, float]):
        self._costs = costs

    def get_position_cost(self, condition_id: str) -> float:
        return self._costs.get(condition_id, 0.0)


class TestGetClusterExposure:
    def test_empty_clusters_returns_zero(self):
        mgr = CorrelationManager()
        assert mgr.get_cluster_exposure("Will X?", _FakeDB({})) == 0.0

    def test_matching_cluster_returns_exposure(self):
        mgr = CorrelationManager()
        positions = [
            _pos("a", "Will Biden resign from presidency?"),
            _pos("b", "Will Biden leave the presidency early?"),
        ]
        mgr.update_clusters(positions)
        db = _FakeDB({"a": 50.0, "b": 30.0})
        exposure = mgr.get_cluster_exposure("Will Biden step down from office?", db)
        assert exposure > 0  # should match the Biden cluster

    def test_no_match_returns_zero(self):
        mgr = CorrelationManager()
        positions = [
            _pos("a", "Will Bitcoin hit one hundred thousand dollars cryptocurrency?"),
        ]
        mgr.update_clusters(positions)
        db = _FakeDB({"a": 50.0})
        exposure = mgr.get_cluster_exposure(
            "Will France win soccer World Cup tournament championship?", db
        )
        assert exposure == 0.0


# === THRESHOLD ===


class TestThreshold:
    def test_default_threshold(self):
        mgr = CorrelationManager()
        assert mgr.similarity_threshold == 0.35
