from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from ..domain.entities import RetrievalIndex
from ..domain.value_objects import VectorHit, VectorRecord


class MilvusStore:
    """向量检索适配器，当前使用进程内存模拟 collection 行为。"""

    def __init__(self) -> None:
        self._collections: defaultdict[str, dict[str, VectorRecord]] = defaultdict(dict)

    def ensure_collections(self, index: RetrievalIndex) -> RetrievalIndex:
        if index.zh_collection_name is None:
            index.zh_collection_name = f"rag_idx_{index.index_id}_zh"
        if index.en_collection_name is None:
            index.en_collection_name = f"rag_idx_{index.index_id}_en"
        return index

    def upsert_entries(self, index: RetrievalIndex, records: list[VectorRecord]) -> None:
        self.ensure_collections(index)
        for record in records:
            collection = self._collection_name(index=index, language=record.language)
            self._collections[collection][str(record.entry_id)] = record

    def hybrid_search(
        self,
        index: RetrievalIndex,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        filters = filters or {}
        hits: list[VectorHit] = []
        for collection_name in filter(None, [index.zh_collection_name, index.en_collection_name]):
            for record in self._collections.get(collection_name, {}).values():
                if not record.is_active:
                    continue
                if not self._match_filters(record=record, filters=filters):
                    continue
                dense_score = self._cosine_similarity(query_vector, record.dense_vector)
                sparse_score = self._lexical_overlap(query_text, record.retrieval_text)
                score = (dense_score + sparse_score) / 2
                hits.append(
                    VectorHit(
                        entry_id=record.entry_id,
                        score=score,
                        dense_score=dense_score,
                        sparse_score=sparse_score,
                    )
                )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def delete_index(self, index: RetrievalIndex) -> None:
        for collection_name in filter(None, [index.zh_collection_name, index.en_collection_name]):
            self._collections.pop(collection_name, None)

    def _collection_name(self, index: RetrievalIndex, language: str) -> str:
        return index.zh_collection_name if language == "zh" else index.en_collection_name

    def _match_filters(self, record: VectorRecord, filters: dict[str, Any]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if key == "file_type" and record.file_type != value:
                return False
            if key == "language" and record.language != value:
                return False
            if record.metadata.get(key) != value:
                return False
        return True

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _lexical_overlap(self, query_text: str, retrieval_text: str) -> float:
        query_terms = set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", query_text))
        target_terms = set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", retrieval_text))
        if not query_terms or not target_terms:
            return 0.0
        return len(query_terms & target_terms) / len(query_terms)
