# LangChain RAG（检索增强生成）

## 学习目标
完成本模块后，你将能够：
1. 理解 RAG 的概念和工作原理
2. 构建完整的 RAG 管道
3. 使用向量数据库存储和检索文档
4. 基于自有数据构建问答系统

---

## 1. 什么是 RAG？

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种让 LLM 访问外部知识的技术。

```
没有 RAG:
  用户问题 → LLM（仅凭训练数据）→ 可能过时的答案

有 RAG:
  用户问题 → [检索相关文档] → LLM（问题 + 文档）→ 准确的答案
```

### 为什么需要 RAG？

| 问题 | 解决方案 |
|------|----------|
| LLM 训练数据有截止日期 | 提供最新的外部文档 |
| LLM 可能"幻觉" | 基于检索到的真实文档回答 |
| 私有知识 LLM 不知道 | 提供公司内部文档 |
| 需要引用来源 | 返回文档来源 |

---

## 2. RAG 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 完整流程                              │
│                                                             │
│  === 离线阶段（一次性）===                                    │
│                                                             │
│  文档 → [分割] → 文本块 → [嵌入] → 向量 → [存储] → 向量数据库  │
│                                                             │
│  === 在线阶段（每次查询）===                                  │
│                                                             │
│  用户问题 → [嵌入] → 查询向量                                 │
│                              ↓                              │
│                    向量数据库 [相似度搜索]                      │
│                              ↓                              │
│                    返回 Top-K 相关文档                         │
│                              ↓                              │
│              [问题 + 相关文档] → LLM → 最终答案                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件

### 3.1 文档加载器（Document Loader）

```python
from langchain_community.document_loaders import (
    TextLoader,       # 加载 .txt 文件
    PyPDFLoader,      # 加载 PDF
    CSVLoader,        # 加载 CSV
    WebBaseLoader,    # 加载网页
    DirectoryLoader,  # 加载整个目录
)

# 示例：加载文本文件
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

# 示例：加载网页
loader = WebBaseLoader("https://example.com")
documents = loader.load()
```

### 3.2 文本分割器（Text Splitter）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 递归字符分割器（推荐）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个块的最大字符数
    chunk_overlap=50,    # 块之间的重叠字符数
    length_function=len,
)

# 分割文档
chunks = splitter.split_documents(documents)
print(f"分割为 {len(chunks)} 个文本块")
```

### 3.3 嵌入模型（Embedding）

将文本转换为向量（数字数组），语义相似的文本向量也相似。

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="your-api-key",
)

# 嵌入单个文本
vector = embeddings.embed_query("Hello World")

# 嵌入多个文本
vectors = embeddings.embed_documents(["文本1", "文本2", "文本3"])
```

### 3.4 向量数据库（Vector Store）

```python
from langchain_community.vectorstores import Chroma

# 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # 持久化目录
)

# 相似度搜索
results = vectorstore.similarity_search("什么是机器学习？", k=3)

# 带分数的搜索
results = vectorstore.similarity_search_with_score("什么是机器学习？", k=3)
```

---

## 4. 构建完整 RAG 管道

### 方法 1：使用 LCEL 构建

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 准备向量数据库（假设已完成）
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. 创建 Prompt
prompt = ChatPromptTemplate.from_template("""根据以下上下文回答问题。
如果上下文中没有相关信息，请说"我不知道"。

上下文:
{context}

问题: {question}

请用中文回答。
""")

# 3. 创建 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 4. 格式化文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. 构建 RAG 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 查询
answer = rag_chain.invoke("什么是机器学习？")
print(answer)
```

### 方法 2：使用 create_retrieval_chain

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 创建文档组合链
document_chain = create_stuff_documents_chain(llm, prompt)

# 创建检索链
rag_chain = create_retrieval_chain(retriever, document_chain)

# 查询
result = rag_chain.invoke({"input": "什么是机器学习？"})
print(result["answer"])
```

---

## 5. 高级检索策略

### 5.1 最大边际相关性（MMR）

```python
# MMR 在相关性和多样性之间取得平衡
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)
```

### 5.2 相似度阈值

```python
# 只返回相似度高于阈值的文档
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.7},
)
```

### 5.3 自查询检索器

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# 支持元数据过滤
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_content_description="文档内容",
    metadata_field_info=[
        AttributeInfo(name="source", description="文档来源", type="string"),
        AttributeInfo(name="date", description="文档日期", type="string"),
    ],
)
```

---

## 6. RAG 评估指标

| 指标 | 说明 |
|------|------|
| 忠实度（Faithfulness） | 答案是否基于检索到的文档 |
| 相关性（Relevance） | 检索到的文档是否与问题相关 |
| 答案相关性 | 答案是否回答了问题 |

---

## 7. 常见问题

### Q: chunk_size 如何选择？
A:
- **小文档/精确问答**：200-500 字符
- **通用场景**：500-1000 字符
- **长文档/摘要**：1000-2000 字符
- `chunk_overlap` 通常为 `chunk_size` 的 10-20%

### Q: 如何选择向量数据库？
A:
| 数据库 | 适用场景 |
|--------|----------|
| Chroma | 本地开发、原型 |
| FAISS | 大规模、高性能 |
| Pinecone | 生产环境、托管 |
| Weaviate | 混合搜索 |

---

## 下一步学习

1. **Memory（记忆）** — 管理多轮对话
2. **LangGraph** — 构建复杂工作流
3. **多模态 RAG** — 处理图片、表格
