"""
NLP-based correlation clustering.
Uses TF-IDF on market questions to group similar markets.
Example: "Will Biden resign?" and "Will Biden leave office?" → same cluster.
Cluster exposure capped at MAX_CLUSTER_EXPOSURE_PCT of bankroll.
"""
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CorrelationManager:
    def __init__(self):
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.clusters: dict[int, list[str]] = {}  # {cluster_id: [condition_ids]}
        self._question_map: dict[str, str] = {}   # {condition_id: question}

    def update_clusters(self, positions: list[dict]) -> None:
        """
        Rebuild clusters from all open positions.
        Called once per cycle if ENABLE_CORRELATION_CLUSTERS=true.

        positions: list of dicts with "condition_id", "market_question"
        """
        if len(positions) < 2:
            self.clusters = {0: [p["condition_id"] for p in positions]} if positions else {}
            return

        self._question_map = {p["condition_id"]: p["market_question"] for p in positions}
        questions = [p["market_question"] for p in positions]
        cids = [p["condition_id"] for p in positions]

        try:
            tfidf_matrix = self.vectorizer.fit_transform(questions)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception:
            # Fallback: each market is its own cluster
            self.clusters = {i: [cid] for i, cid in enumerate(cids)}
            return

        # Greedy clustering: assign each market to first cluster with similarity > threshold
        self.clusters = {}
        assigned: set[str] = set()
        cluster_id = 0

        for i in range(len(cids)):
            if cids[i] in assigned:
                continue
            cluster = [cids[i]]
            assigned.add(cids[i])
            for j in range(i + 1, len(cids)):
                if cids[j] not in assigned and sim_matrix[i, j] >= self.similarity_threshold:
                    cluster.append(cids[j])
                    assigned.add(cids[j])
            self.clusters[cluster_id] = cluster
            cluster_id += 1

    def get_cluster_exposure(self, question: str, db) -> float:
        """
        Return total USD exposure in the cluster that this question would join.
        If no cluster match → 0 (new cluster).
        """
        if not self.clusters:
            return 0.0

        # Find which cluster this question is most similar to
        for cid_list in self.clusters.values():
            existing_questions = [self._question_map.get(c, "") for c in cid_list]
            if not existing_questions:
                continue
            try:
                all_qs = existing_questions + [question]
                tfidf = self.vectorizer.transform(all_qs)
                sims = cosine_similarity(tfidf[-1:], tfidf[:-1])[0]
                if np.max(sims) >= self.similarity_threshold:
                    # This question would join this cluster
                    exposure = sum(
                        db.get_position_cost(c) for c in cid_list
                    )
                    return exposure
            except Exception:
                continue
        return 0.0
