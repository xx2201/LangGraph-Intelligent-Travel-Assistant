from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.config import (
    LONG_TERM_MEMORY_BACKEND,
    LONG_TERM_MEMORY_DB_PATH,
    MILVUS_COLLECTION_NAME,
    MILVUS_TOKEN,
    MILVUS_URI,
)

try:  # pragma: no cover - optional dependency
    from pymilvus import DataType, MilvusClient
except Exception:  # pragma: no cover - optional dependency
    MilvusClient = None
    DataType = None


EMBEDDING_DIMENSION = 64
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize_text(text: str) -> list[str]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return []
    return TOKEN_PATTERN.findall(cleaned.lower())


def embed_text(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    tokens = tokenize_text(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimension
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        weight = 1.0 + (digest[2] / 255.0)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(l * r for l, r in zip(left, right))


@dataclass(slots=True)
class MemoryRecord:
    id: str
    scope: str
    memory_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class BaseLongTermMemoryStore:
    def search(
        self,
        *,
        scope: str,
        query: str,
        top_k: int,
        memory_types: Iterable[str] | None = None,
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    def upsert(
        self,
        *,
        scope: str,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> str:
        raise NotImplementedError

    def delete_scope(self, scope: str) -> None:
        raise NotImplementedError

    async def asearch(
        self,
        *,
        scope: str,
        query: str,
        top_k: int,
        memory_types: Iterable[str] | None = None,
    ) -> list[MemoryRecord]:
        return await asyncio.to_thread(
            self.search,
            scope=scope,
            query=query,
            top_k=top_k,
            memory_types=memory_types,
        )

    async def aupsert(
        self,
        *,
        scope: str,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.upsert,
            scope=scope,
            memory_type=memory_type,
            content=content,
            metadata=metadata,
            record_id=record_id,
        )

    async def adelete_scope(self, scope: str) -> None:
        await asyncio.to_thread(self.delete_scope, scope)


class NoOpLongTermMemoryStore(BaseLongTermMemoryStore):
    def search(
        self,
        *,
        scope: str,
        query: str,
        top_k: int,
        memory_types: Iterable[str] | None = None,
    ) -> list[MemoryRecord]:
        return []

    def upsert(
        self,
        *,
        scope: str,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> str:
        return record_id or ""

    def delete_scope(self, scope: str) -> None:
        return None


class SQLiteLongTermMemoryStore(BaseLongTermMemoryStore):
    def __init__(self, db_path: str = LONG_TERM_MEMORY_DB_PATH) -> None:
        self.db_path = str(db_path)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_long_term_memories_scope
                ON long_term_memories(scope)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_long_term_memories_scope_type
                ON long_term_memories(scope, memory_type)
                """
            )
            connection.commit()

    def search(
        self,
        *,
        scope: str,
        query: str,
        top_k: int,
        memory_types: Iterable[str] | None = None,
    ) -> list[MemoryRecord]:
        query_vector = embed_text(query)
        clauses = ["scope = ?"]
        params: list[Any] = [scope]
        allowed_types = list(memory_types or [])
        if allowed_types:
            placeholders = ",".join("?" for _ in allowed_types)
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(allowed_types)
        sql = (
            "SELECT id, scope, memory_type, content, metadata_json, embedding_json "
            "FROM long_term_memories WHERE " + " AND ".join(clauses)
        )
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        scored_records: list[MemoryRecord] = []
        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query_vector, embedding)
            if score <= 0:
                continue
            scored_records.append(
                MemoryRecord(
                    id=row["id"],
                    scope=row["scope"],
                    memory_type=row["memory_type"],
                    content=row["content"],
                    metadata=json.loads(row["metadata_json"]),
                    score=score,
                )
            )
        scored_records.sort(key=lambda record: record.score, reverse=True)
        return scored_records[:top_k]

    def upsert(
        self,
        *,
        scope: str,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> str:
        normalized_content = " ".join((content or "").strip().split())
        if not normalized_content:
            return record_id or ""
        now = utc_now_iso()
        resolved_record_id = record_id or hashlib.sha1(
            f"{scope}:{memory_type}:{normalized_content}".encode("utf-8")
        ).hexdigest()
        payload = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        embedding = json.dumps(embed_text(normalized_content))
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM long_term_memories WHERE id = ?",
                (resolved_record_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            connection.execute(
                """
                INSERT OR REPLACE INTO long_term_memories (
                    id, scope, memory_type, content, metadata_json, embedding_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_record_id,
                    scope,
                    memory_type,
                    normalized_content,
                    payload,
                    embedding,
                    created_at,
                    now,
                ),
            )
            connection.commit()
        return resolved_record_id

    def delete_scope(self, scope: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM long_term_memories WHERE scope = ?",
                (scope,),
            )
            connection.commit()


class MilvusLongTermMemoryStore(BaseLongTermMemoryStore):
    def __init__(
        self,
        *,
        uri: str = MILVUS_URI,
        token: str = MILVUS_TOKEN,
        collection_name: str = MILVUS_COLLECTION_NAME,
    ) -> None:
        if MilvusClient is None:  # pragma: no cover - optional dependency
            raise RuntimeError("pymilvus is not installed")
        if not uri:  # pragma: no cover - configuration guard
            raise RuntimeError("MILVUS_URI is required when LONG_TERM_MEMORY_BACKEND=milvus")
        self.collection_name = collection_name
        self.client = MilvusClient(uri=uri, token=token or None)
        self._initialize()

    def _initialize(self) -> None:
        if self.client.has_collection(collection_name=self.collection_name):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="scope", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="memory_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIMENSION,
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="AUTOINDEX",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def search(
        self,
        *,
        scope: str,
        query: str,
        top_k: int,
        memory_types: Iterable[str] | None = None,
    ) -> list[MemoryRecord]:
        filter_parts = [f'scope == "{scope}"']
        allowed_types = list(memory_types or [])
        if allowed_types:
            quoted = ", ".join(f'"{memory_type}"' for memory_type in allowed_types)
            filter_parts.append(f"memory_type in [{quoted}]")
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embed_text(query)],
            filter=" and ".join(filter_parts),
            limit=top_k,
            output_fields=["scope", "memory_type", "content", "metadata_json"],
        )
        flattened = results[0] if results else []
        records: list[MemoryRecord] = []
        for item in flattened:
            entity = item.get("entity", item)
            metadata_json = entity.get("metadata_json", "{}")
            records.append(
                MemoryRecord(
                    id=str(entity.get("id", "")),
                    scope=entity.get("scope", scope),
                    memory_type=entity.get("memory_type", ""),
                    content=entity.get("content", ""),
                    metadata=json.loads(metadata_json) if metadata_json else {},
                    score=float(item.get("distance", 0.0)),
                )
            )
        return records

    def upsert(
        self,
        *,
        scope: str,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> str:
        normalized_content = " ".join((content or "").strip().split())
        if not normalized_content:
            return record_id or ""
        resolved_record_id = record_id or hashlib.sha1(
            f"{scope}:{memory_type}:{normalized_content}".encode("utf-8")
        ).hexdigest()
        self.client.upsert(
            collection_name=self.collection_name,
            data=[
                {
                    "id": resolved_record_id,
                    "scope": scope,
                    "memory_type": memory_type,
                    "content": normalized_content,
                    "metadata_json": json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    "vector": embed_text(normalized_content),
                }
            ],
        )
        return resolved_record_id

    def delete_scope(self, scope: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'scope == "{scope}"',
        )


class LongTermMemoryManager:
    def __init__(self, store: BaseLongTermMemoryStore, top_k: int = 3) -> None:
        self.store = store
        self.top_k = top_k

    def recall(
        self,
        *,
        scope: str,
        query: str,
        memory_types: Iterable[str] | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        if not scope or not query.strip():
            return []
        records = self.store.search(
            scope=scope,
            query=query,
            top_k=top_k or self.top_k,
            memory_types=memory_types,
        )
        return [record.content for record in records]

    async def arecall(
        self,
        *,
        scope: str,
        query: str,
        memory_types: Iterable[str] | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        if not scope or not query.strip():
            return []
        records = await self.store.asearch(
            scope=scope,
            query=query,
            top_k=top_k or self.top_k,
            memory_types=memory_types,
        )
        return [record.content for record in records]

    def remember(
        self,
        *,
        scope: str,
        conversation_summary: str,
        task_memory: dict[str, Any],
    ) -> None:
        if not scope:
            return
        if conversation_summary.strip():
            self.store.upsert(
                scope=scope,
                memory_type="summary",
                content=conversation_summary,
                metadata={"kind": "conversation_summary"},
                record_id=f"{scope}:summary",
            )
        current_goal = str(task_memory.get("current_goal", "")).strip()
        if current_goal:
            self.store.upsert(
                scope=scope,
                memory_type="goal",
                content=current_goal,
                metadata={"kind": "current_goal"},
                record_id=f"{scope}:goal",
            )
        for preference in task_memory.get("user_preferences", []):
            cleaned = " ".join(str(preference).split())
            if not cleaned:
                continue
            digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
            self.store.upsert(
                scope=scope,
                memory_type="preference",
                content=cleaned,
                metadata={"kind": "user_preference"},
                record_id=f"{scope}:preference:{digest}",
            )

    async def aremember(
        self,
        *,
        scope: str,
        conversation_summary: str,
        task_memory: dict[str, Any],
    ) -> None:
        await asyncio.to_thread(
            self.remember,
            scope=scope,
            conversation_summary=conversation_summary,
            task_memory=task_memory,
        )

    def delete_scope(self, scope: str) -> None:
        self.store.delete_scope(scope)

    async def adelete_scope(self, scope: str) -> None:
        await self.store.adelete_scope(scope)


def build_long_term_memory_manager(
    *,
    backend: str = LONG_TERM_MEMORY_BACKEND,
    db_path: str = LONG_TERM_MEMORY_DB_PATH,
    uri: str = MILVUS_URI,
    token: str = MILVUS_TOKEN,
    collection_name: str = MILVUS_COLLECTION_NAME,
    top_k: int = 3,
) -> LongTermMemoryManager:
    normalized_backend = (backend or "sqlite").strip().lower()
    store: BaseLongTermMemoryStore
    if normalized_backend == "milvus":
        try:
            store = MilvusLongTermMemoryStore(
                uri=uri,
                token=token,
                collection_name=collection_name,
            )
        except Exception:
            store = SQLiteLongTermMemoryStore(db_path=db_path)
    elif normalized_backend == "none":
        store = NoOpLongTermMemoryStore()
    else:
        store = SQLiteLongTermMemoryStore(db_path=db_path)
    return LongTermMemoryManager(store=store, top_k=top_k)
