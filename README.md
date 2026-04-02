# ModularRagEngine

面向 Python 后端系统的分层 RAG 引擎

## 👋 项目介绍

`ModularRagEngine` 是一个面向 Python 后端系统的分层 RAG 组件，提供统一的文档接入、分块、索引构建、检索与上下文组装能力。

它适合嵌入法律问答助手、知识库检索、合同审查、内部文档问答等垂直领域系统，让宿主系统在不耦合底层实现细节的前提下，直接获得完整的 RAG 数据流能力。

<p align="center">
  <img src="./images/setup.png" width="980" alt="ModularRagEngine workflow" />
</p>

## 🔥 功能与特色

- 分层结构清晰：采用 `api -> application -> domain <- infrastructure` 四层架构。
- 接入方式统一：同时支持文件入库与宿主系统直接传入已解析文档。
- 作用域隔离明确：使用 `namespace` 管理独立检索空间。
- 检索链路完整：覆盖 `ingest -> index -> search -> context assembly`。
- 基础设施可替换：当前默认使用内存实现，后续可平滑替换为 PostgreSQL / Milvus。
- 旧代码已归档：历史实现放在 `backup/`，主仓库结构保持干净。

## 🚀 快速开始

### 1. 创建引擎

```python
from ModularRagEngine import RAGEngine

engine = RAGEngine()
```

### 2. 写入文档

```python
from ModularRagEngine.application.dto import IngestDocumentsRequest, InputDocument

engine.ingest_documents(
    IngestDocumentsRequest(
        namespace_key="legal-case-001",
        documents=[
            InputDocument(
                external_doc_id="doc-001",
                file_name="case.md",
                file_type="md",
                parsed_md_content=(
                    "# Case Background\n\n"
                    "The dispute focuses on liquidated damages.\n\n"
                    "## Judgment\n\n"
                    "The court supports the plaintiff's claim."
                ),
            )
        ],
    )
)
```

### 3. 执行检索

```python
from ModularRagEngine.application.dto import SearchRequest

result = engine.search(
    SearchRequest(
        namespace_key="legal-case-001",
        query="liquidated damages",
        top_k_recall=8,
        top_k_rerank=5,
        top_k_context=3,
    )
)

print(result.llm_context)
```

## 🧩 对外 API

- `RAGEngine.ingest_files()`
- `RAGEngine.ingest_documents()`
- `RAGEngine.rebuild_index()`
- `RAGEngine.search()`

请求与响应模型位于 `application/dto.py`，当前入口统一由 `RAGEngine` 暴露。

## 🏗️ 目录结构

```text
ModularRagEngine/
├── api/                # 对外 Facade
├── application/        # 应用编排层
├── domain/             # 领域模型与端口
├── infrastructure/     # 基础设施实现
├── utils/              # OCR / embedding 等外部能力适配
├── backup/             # 归档的旧实现
├── images/             # README 展示资源
├── config.py           # 配置入口
└── __init__.py         # 包根导出
```

## 🙋‍♂️ 贡献

欢迎提交 Issue 或 Pull Request 改进项目结构、接口设计、基础设施适配器和文档质量。

## 📝 许可证

按仓库后续约定补充。
