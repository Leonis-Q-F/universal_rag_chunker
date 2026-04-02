from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from uuid import UUID

from ..domain.entities import ChildBlock, IndexEntry, Namespace, ParentChunk, RetrievalIndex, SourceDocument
from ..domain.exceptions import NamespaceNotFoundError
from ..domain.value_objects import ChunkBundle


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStore:
    """内容层与索引元数据存储，当前为进程内实现。"""

    def __init__(self) -> None:
        self._namespaces_by_id: dict[UUID, Namespace] = {}
        self._namespaces_by_key: dict[str, UUID] = {}
        self._documents: dict[UUID, SourceDocument] = {}
        self._doc_keys: dict[tuple[UUID, str], UUID] = {}
        self._parent_chunks: dict[UUID, ParentChunk] = {}
        self._child_blocks: dict[UUID, ChildBlock] = {}
        self._parents_by_doc: defaultdict[UUID, list[UUID]] = defaultdict(list)
        self._blocks_by_doc: defaultdict[UUID, list[UUID]] = defaultdict(list)
        self._indexes: dict[UUID, RetrievalIndex] = {}
        self._indexes_by_namespace: defaultdict[UUID, list[UUID]] = defaultdict(list)
        self._entries: dict[UUID, IndexEntry] = {}
        self._entries_by_index: defaultdict[UUID, list[UUID]] = defaultdict(list)

    def ensure_namespace(self, namespace_key: str, namespace_name: str | None = None) -> Namespace:
        if namespace_key in self._namespaces_by_key:
            namespace = self._namespaces_by_id[self._namespaces_by_key[namespace_key]]
            namespace.updated_at = utc_now()
            return namespace

        namespace = Namespace(namespace_key=namespace_key, namespace_name=namespace_name or namespace_key)
        self._namespaces_by_id[namespace.namespace_id] = namespace
        self._namespaces_by_key[namespace.namespace_key] = namespace.namespace_id
        return namespace

    def get_namespace(self, namespace_id: UUID | None = None, namespace_key: str | None = None) -> Namespace:
        if namespace_id is not None and namespace_id in self._namespaces_by_id:
            return self._namespaces_by_id[namespace_id]
        if namespace_key and namespace_key in self._namespaces_by_key:
            return self._namespaces_by_id[self._namespaces_by_key[namespace_key]]
        raise NamespaceNotFoundError("namespace 不存在。")

    def upsert_source_documents(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        stored: list[SourceDocument] = []
        for document in documents:
            key = (document.namespace_id, document.external_doc_id or document.content_sha256)
            if key in self._doc_keys:
                doc_id = self._doc_keys[key]
                existing = self._documents[doc_id]
                existing.file_name = document.file_name
                existing.file_type = document.file_type
                existing.source_uri = document.source_uri
                existing.language = document.language
                existing.content_sha256 = document.content_sha256
                existing.parser_name = document.parser_name
                existing.parser_version = document.parser_version
                existing.metadata = dict(document.metadata)
                existing.parsed_md_content = document.parsed_md_content
                existing.updated_at = utc_now()
                stored.append(existing)
                continue

            self._documents[document.doc_id] = document
            self._doc_keys[key] = document.doc_id
            stored.append(document)
        return stored

    def replace_document_chunks(self, bundle: ChunkBundle) -> ChunkBundle:
        doc_id = bundle.source_document.doc_id
        for parent_id in self._parents_by_doc.get(doc_id, []):
            self._parent_chunks[parent_id].is_active = False
            self._parent_chunks[parent_id].deleted_at = utc_now()
        for block_id in self._blocks_by_doc.get(doc_id, []):
            self._child_blocks[block_id].is_active = False
            self._child_blocks[block_id].deleted_at = utc_now()

        self._parents_by_doc[doc_id] = []
        self._blocks_by_doc[doc_id] = []

        for parent_chunk in bundle.parent_chunks:
            self._parent_chunks[parent_chunk.parent_id] = parent_chunk
            self._parents_by_doc[doc_id].append(parent_chunk.parent_id)
        for child_block in bundle.child_blocks:
            self._child_blocks[child_block.block_id] = child_block
            self._blocks_by_doc[doc_id].append(child_block.block_id)
        return bundle

    def list_child_blocks(self, namespace_id: UUID, doc_ids: list[UUID] | None = None) -> list[ChildBlock]:
        result: list[ChildBlock] = []
        allowed_doc_ids = set(doc_ids or [])
        for block in self._child_blocks.values():
            if not block.is_active or block.namespace_id != namespace_id:
                continue
            if allowed_doc_ids and block.doc_id not in allowed_doc_ids:
                continue
            result.append(block)
        result.sort(key=lambda item: (str(item.doc_id), item.child_index))
        return result

    def create_index(self, index: RetrievalIndex) -> RetrievalIndex:
        self._indexes[index.index_id] = index
        self._indexes_by_namespace[index.namespace_id].append(index.index_id)
        return index

    def get_active_index(self, namespace_id: UUID) -> RetrievalIndex | None:
        for index_id in self._indexes_by_namespace.get(namespace_id, []):
            index = self._indexes[index_id]
            if index.is_active and index.deleted_at is None:
                return index
        return None

    def list_indexes(self, namespace_id: UUID) -> list[RetrievalIndex]:
        return [self._indexes[index_id] for index_id in self._indexes_by_namespace.get(namespace_id, []) if self._indexes[index_id].deleted_at is None]

    def save_index_entries(self, entries: list[IndexEntry]) -> list[IndexEntry]:
        saved: list[IndexEntry] = []
        for entry in entries:
            entry.vector_primary_key = str(entry.entry_id)
            entry.updated_at = utc_now()
            self._entries[entry.entry_id] = entry
            self._entries_by_index[entry.index_id].append(entry.entry_id)
            saved.append(entry)
        return saved

    def deactivate_index_entries(self, index_id: UUID, doc_ids: list[UUID] | None = None) -> None:
        allowed_doc_ids = set(doc_ids or [])
        for entry_id in self._entries_by_index.get(index_id, []):
            entry = self._entries[entry_id]
            if allowed_doc_ids and entry.doc_id not in allowed_doc_ids:
                continue
            entry.is_active = False
            entry.deleted_at = utc_now()

    def get_index_entries(self, entry_ids: list[UUID]) -> list[IndexEntry]:
        result = []
        for entry_id in entry_ids:
            entry = self._entries.get(entry_id)
            if entry is not None and entry.is_active and entry.deleted_at is None:
                result.append(entry)
        return result

    def get_parent_chunks(self, parent_ids: list[UUID]) -> list[ParentChunk]:
        result: list[ParentChunk] = []
        for parent_id in parent_ids:
            parent_chunk = self._parent_chunks.get(parent_id)
            if parent_chunk is not None and parent_chunk.is_active and parent_chunk.deleted_at is None:
                result.append(parent_chunk)
        return result

    def get_parent_chunk_window(self, parent_id: UUID, window: int) -> list[ParentChunk]:
        parent = self._parent_chunks[parent_id]
        parent_ids = self._parents_by_doc.get(parent.doc_id, [])
        siblings = [self._parent_chunks[item] for item in parent_ids if self._parent_chunks[item].is_active and self._parent_chunks[item].deleted_at is None]
        siblings.sort(key=lambda item: item.chunk_index)
        start = max(parent.chunk_index - window, 0)
        end = parent.chunk_index + window + 1
        return [chunk for chunk in siblings if start <= chunk.chunk_index < end]

    def activate_index(self, index_id: UUID) -> RetrievalIndex:
        target = self._indexes[index_id]
        for sibling_id in self._indexes_by_namespace.get(target.namespace_id, []):
            sibling = self._indexes[sibling_id]
            if sibling.index_id == target.index_id:
                sibling.is_active = True
                sibling.status = "ready"
                sibling.activated_at = utc_now()
                sibling.deleted_at = None
            else:
                sibling.is_active = False
                if sibling.status != "failed":
                    sibling.status = "retired"

        for entry_id, entry in self._entries.items():
            if self._indexes[entry.index_id].namespace_id != target.namespace_id:
                continue
            entry.is_active = entry.index_id == target.index_id and entry.deleted_at is None
        return target

    def update_index_status(self, index_id: UUID, status: str, is_active: bool | None = None) -> RetrievalIndex:
        index = self._indexes[index_id]
        index.status = status
        index.updated_at = utc_now()
        if is_active is not None:
            index.is_active = is_active
        return index
