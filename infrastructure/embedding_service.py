from __future__ import annotations

import hashlib
import math
import re


class EmbeddingService:
    """默认的本地确定性 embedding 服务，便于无外部依赖运行。"""

    provider_name = "builtin"
    model_name = "hash-embedding-v1"
    dimension = 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dimension
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokenize(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text)
