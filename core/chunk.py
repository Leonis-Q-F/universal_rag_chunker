from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

from universal_rag_chunker.models.schemes import ParentChunk, SourceDocument

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - 兼容旧版 LangChain 安装方式
    try:
        from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - 运行时依赖保护
        MarkdownHeaderTextSplitter = None
        RecursiveCharacterTextSplitter = None

try:
    import tiktoken
except ImportError:  # pragma: no cover - 可选依赖
    tiktoken = None


@dataclass(slots=True)
class SplitSection:
    content: str
    metadata: dict[str, Any]


class StructuralSplitConfig(BaseModel):
    """一级父块切分配置。"""

    max_tokens: int = Field(default=1200, ge=50, description="每个父块的目标长度上限。")
    chunk_overlap: int = Field(default=100, ge=0, description="仅在超长分段再次切分时使用的重叠长度。")
    tokenizer_encoding: str = Field(default="cl100k_base", description="tiktoken 使用的编码名称。")
    strip_headers: bool = Field(default=False, description="按标题切分时，是否从正文中移除 Markdown 标题行。")
    headers_to_split_on: tuple[tuple[str, str], ...] = Field(
        default=(("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")),
        description="一级结构切分时使用的 Markdown 标题层级。",
    )

    def model_post_init(self, __context: Any) -> None:
        """校验配置是否合法，避免重叠长度大于等于目标长度。"""
        if self.chunk_overlap >= self.max_tokens:
            raise ValueError("chunk_overlap 必须小于 max_tokens")


class TokenCounter:
    """用于长度判断和 metadata 记录的轻量计数器。"""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """初始化计数器，优先使用 tiktoken 进行真实长度统计。"""
        self._encoding = None
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                self._encoding = None

    def count(self, text: str) -> int:
        """统计文本长度，优先返回 token 数，缺少依赖时退化为近似值。"""
        text = text.strip()
        if not text:
            return 0

        if self._encoding is not None:
            return len(self._encoding.encode(text))

        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text))


class MarkdownStructureSplitter(BaseModel):
    """基于 LangChain 文本切分能力的一级结构切分器。"""

    config: StructuralSplitConfig = Field(default_factory=StructuralSplitConfig)

    _token_counter: TokenCounter = PrivateAttr()
    _header_splitter: Any = PrivateAttr()
    _length_splitter: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """初始化一级切分所需的标题切分器、长度切分器和计数器。"""
        if MarkdownHeaderTextSplitter is None or RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "缺少 LangChain 文本切分依赖，请安装 `langchain-text-splitters` "
                "或兼容版本的 `langchain`。"
            )

        self._token_counter = TokenCounter(self.config.tokenizer_encoding)
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(self.config.headers_to_split_on),
            strip_headers=self.config.strip_headers,
        )
        self._length_splitter = self._build_length_splitter()

    def split_document(self, doc: SourceDocument) -> list[ParentChunk]:
        """对 SourceDocument 执行一级切分，并输出 ParentChunk 列表。"""
        sections = self.split_markdown(doc.parsed_md_content)
        return self._to_parent_chunks(doc, sections)

    def split_markdown(self, markdown: str) -> list[SplitSection]:
        """对 Markdown 文本执行一级切分，返回内部使用的分段结果。"""
        sections = self._split_by_headers(markdown)
        return self._split_oversized_sections(sections)

    def _split_by_headers(self, markdown: str) -> list[SplitSection]:
        """先按 Markdown 标题切分文本，得到结构化的初始分段。"""
        markdown = markdown.strip()
        if not markdown:
            return []

        documents = self._header_splitter.split_text(markdown)
        if not documents:
            return [SplitSection(content=markdown, metadata={})]

        sections = [
            SplitSection(content=document.page_content.strip(), metadata=dict(document.metadata or {}))
            for document in documents
            if document.page_content.strip()
        ]
        return sections or [SplitSection(content=markdown, metadata={})]

    def _split_oversized_sections(self, sections: list[SplitSection]) -> list[SplitSection]:
        """对超长分段做长度兜底切分，并在需要时补回标题前缀。"""
        expanded: list[SplitSection] = []

        for section in sections:
            if self._token_counter.count(section.content) <= self.config.max_tokens:
                expanded.append(section)
                continue

            chunks = [chunk.strip() for chunk in self._length_splitter.split_text(section.content) if chunk.strip()]
            if len(chunks) <= 1:
                expanded.append(section)
                continue

            header_prefix = self._render_header_prefix(section.metadata)
            total = len(chunks)

            for index, chunk in enumerate(chunks):
                content = chunk
                if header_prefix and (index > 0 or self.config.strip_headers):
                    content = f"{header_prefix}\n\n{chunk}"

                metadata = {
                    **section.metadata,
                    "split_part": index,
                    "split_total": total,
                }
                expanded.append(SplitSection(content=content, metadata=metadata))

        return expanded

    def _to_parent_chunks(
        self,
        doc: SourceDocument,
        sections: list[SplitSection],
    ) -> list[ParentChunk]:
        """把内部切分结果转换成统一的 ParentChunk 数据结构。"""
        return [
            ParentChunk(
                parent_id=uuid4(),
                doc_id=doc.doc_id,
                content=section.content,
                metadata={
                    "file_name": doc.file_name,
                    "file_type": doc.file_type.value,
                    "header_path": self._header_path(section.metadata),
                    "heading_level": self._heading_level(section.metadata),
                    "token_count": self._token_counter.count(section.content),
                    **section.metadata,
                },
                chunk_index=index,
            )
            for index, section in enumerate(sections)
        ]

    def _build_length_splitter(self) -> Any:
        """构建按长度兜底切分的工具，优先使用基于 tiktoken 的实现。"""
        separators = ["\n\n", "\n", "。", "！", "？", ".", "；", ";", "，", ",", " ", ""]

        try:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.config.tokenizer_encoding,
                chunk_size=self.config.max_tokens,
                chunk_overlap=self.config.chunk_overlap,
                separators=separators,
            )
        except Exception:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.max_tokens,
                chunk_overlap=self.config.chunk_overlap,
                separators=separators,
            )

    def _header_path(self, metadata: dict[str, Any]) -> list[str]:
        """从标题元数据中提取层级路径。"""
        path: list[str] = []
        for _, key in self.config.headers_to_split_on:
            value = metadata.get(key)
            if value:
                path.append(str(value))
        return path

    def _heading_level(self, metadata: dict[str, Any]) -> int | None:
        """根据标题路径长度推导当前分段的标题层级。"""
        level = len(self._header_path(metadata))
        return level or None

    def _render_header_prefix(self, metadata: dict[str, Any]) -> str:
        """把标题元数据重新渲染成 Markdown 标题前缀，用于补回上下文。"""
        lines: list[str] = []
        for header_mark, key in self.config.headers_to_split_on:
            value = metadata.get(key)
            if value:
                lines.append(f"{header_mark} {value}")
        return "\n".join(lines).strip()


__all__ = [
    "MarkdownStructureSplitter",
    "SplitSection",
    "StructuralSplitConfig",
    "TokenCounter",
]
