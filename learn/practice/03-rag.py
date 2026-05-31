#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 3: RAG（检索增强生成）
学习目标：
1. 理解 RAG 的工作原理
2. 学会文本切分（Text Splitter）
3. 学会文本向量化（Embedding）
4. 构建一个完整的 RAG 问答系统
"""

import os
import sys
import io
import json
import math

# 修复 Windows 终端编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# ============================================================
# 核心概念 1: 文本切分（Text Splitter）
# ============================================================

def demo_text_splitter():
    """演示文本切分"""
    print("=" * 60)
    print("示例 1: 文本切分（Text Splitter）")
    print("=" * 60)

    text = (
        "LangChain 是一个用于构建 LLM 应用的框架。"
        "它提供了标准化的接口来调用不同的模型。"
        "RAG 是检索增强生成技术，能够基于外部知识库回答问题。"
        "向量数据库用于存储和检索文本嵌入。"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        separators=["。", "！", "？", "\n"],
    )

    chunks = splitter.split_text(text)
    print(f"原始文本长度: {len(text)} 字符")
    print(f"切分后片段数: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  片段 {i + 1} ({len(chunk)} 字符): {chunk}")
    print()


# ============================================================
# 核心概念 2: 文本向量化（Embedding）— 使用本地模型
# ============================================================

def demo_embedding():
    """演示文本向量化原理（模拟）"""
    print("=" * 60)
    print("示例 2: 文本向量化（Embedding）原理")
    print("=" * 60)

    print("""
Embedding 是将文本转换为数值向量的过程。
由于 HuggingFace 在国内无法访问，这里用模拟数据演示原理。

实际使用中:
  - OpenAI: text-embedding-3-small (1536维)
  - 本地:   sentence-transformers/all-MiniLM-L6-v2 (384维)
  - 中文:   BAAI/bge-small-zh (512维)

模拟向量（实际向量通常是 384~1568 维的浮点数）:
""")

    # 模拟向量（用简单的哈希模拟）
    def mock_embed(text: str) -> list[float]:
        """模拟嵌入：基于字符哈希生成伪向量"""
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 16, 2)]

    texts = [
        "LangChain 是一个 LLM 框架",
        "Python 是一种编程语言",
        "RAG 是检索增强生成技术",
    ]

    vectors = [mock_embed(t) for t in texts]

    print(f"文本向量化结果（模拟 8 维向量）:")
    for text, vec in zip(texts, vectors):
        print(f"  '{text}'")
        print(f"    → [{', '.join(f'{v:.3f}' for v in vec)}]")

    # 计算相似度
    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0

    sim_01 = cosine_similarity(vectors[0], vectors[1])
    sim_02 = cosine_similarity(vectors[0], vectors[2])
    print(f"\n余弦相似度（模拟）:")
    print(f"  '{texts[0]}' vs '{texts[1]}': {sim_01:.4f}")
    print(f"  '{texts[0]}' vs '{texts[2]}': {sim_02:.4f}")
    print(f"  → 语义越相似，向量越接近")
    print()
    return None  # 返回 None，后续使用关键词匹配


# ============================================================
# 核心概念 3: RAG 工作原理
# ============================================================

def demo_rag_concept():
    """展示 RAG 工作原理"""
    print("=" * 60)
    print("示例 3: RAG 工作原理")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────────┐
│                    RAG 完整流程                               │
│                                                             │
│  ① 离线建库（Indexing）                                      │
│     文档 → 切分 → 向量化 → 存入向量数据库                      │
│                                                             │
│  ② 在线检索（Retrieval + Generation）                        │
│     用户问题 ──→ 向量化 ──→ 向量数据库检索                      │
│                                    │                        │
│                                    ▼                        │
│                              Top-K 相关文档                   │
│                                    │                        │
│                                    ▼                        │
│     问题 + 相关文档 ──→ LLM ──→ 最终回答                      │
└─────────────────────────────────────────────────────────────┘

关键优势:
  ✅ 不需要微调模型
  ✅ 知识可以实时更新
  ✅ 回答有据可查（可溯源）
  ✅ 减少模型幻觉
    """)
    print()


# ============================================================
# 核心概念 4: 完整 RAG 系统
# ============================================================

# 模拟知识库
KNOWLEDGE_BASE = [
    {
        "title": "LangChain 简介",
        "content": (
            "LangChain 是一个用于构建大语言模型（LLM）应用的框架。"
            "它提供了标准化的接口来调用 OpenAI、Anthropic、Google 等不同模型。"
            "核心组件包括：Models、Prompts、Chains、Agents、Memory、Retrieval。"
            "LangChain 使用 LCEL（LangChain Expression Language）来构建链式调用。"
        ),
    },
    {
        "title": "RAG 技术",
        "content": (
            "RAG（Retrieval-Augmented Generation）是检索增强生成技术。"
            "它通过检索外部知识库来增强模型的回答能力。"
            "RAG 的核心流程：文档切分 → 向量化 → 存储 → 检索 → 生成。"
            "常用的向量数据库有 Chroma、Pinecone、Weaviate、Qdrant 等。"
            "RAG 的优势是不需要微调模型，知识可以实时更新。"
        ),
    },
    {
        "title": "向量数据库",
        "content": (
            "向量数据库专门用于存储和检索高维向量数据。"
            "它通过计算向量之间的相似度（如余弦相似度）来查找相关文档。"
            "常见的向量索引算法有 HNSW、IVF、PQ 等。"
            "Chroma 是一个轻量级的开源向量数据库，适合本地开发。"
            "Pinecone 是一个云端的向量数据库服务，适合生产环境。"
        ),
    },
    {
        "title": "Embedding 模型",
        "content": (
            "Embedding 模型将文本转换为高维向量（通常是 384~1536 维）。"
            "相似的文本在向量空间中距离更近。"
            "常用的开源 Embedding 模型有："
            "sentence-transformers/all-MiniLM-L6-v2（英文，384维）、"
            "BAAI/bge-small-zh（中文，512维）、"
            "text-embedding-ada-002（OpenAI，1536维）。"
            "选择 Embedding 模型时需要考虑语言、维度、速度等因素。"
        ),
    },
    {
        "title": "文本切分策略",
        "content": (
            "文本切分是 RAG 的关键步骤，直接影响检索质量。"
            "常用的切分策略有："
            "1. 固定长度切分（简单但可能切断语义）"
            "2. 递归字符切分（按段落、句子、单词逐级切分）"
            "3. 语义切分（基于语义相似度）"
            "切分参数：chunk_size（片段大小）、chunk_size_overlap（重叠大小）。"
            "一般建议 chunk_size=200~500，overlap=20~50。"
        ),
    },
]


def demo_full_rag(embeddings):
    """完整 RAG 系统演示"""
    print("=" * 60)
    print("示例 4: 完整 RAG 系统")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")

    llm_kwargs = {
        "model": model_name,
        "temperature": 0,
        "api_key": api_key,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = ChatOpenAI(**llm_kwargs)

    # 如果没有 embedding 模型，用简单的关键词匹配
    if embeddings is None:
        print("⚠️  无 Embedding 模型，使用关键词匹配代替向量检索\n")

        def simple_retrieve(query: str, k: int = 2) -> str:
            """基于关键词重叠的简单检索"""
            query_words = set(query.lower().split())
            results = []
            for item in KNOWLEDGE_BASE:
                content_words = set(item["content"].lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    results.append((overlap, item))
            results.sort(key=lambda x: x[0], reverse=True)
            top_k = [item for _, item in results[:k]]
            return "\n\n".join(
                f"[{item['title']}]\n{item['content']}" for item in top_k
            )

        # 构建 RAG 链
        template = """基于以下上下文回答问题。如果上下文中没有相关信息，请如实说明。

上下文:
{context}

问题: {question}

请用中文回答:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_input(inputs):
            return {
                "context": simple_retrieve(inputs["question"]),
                "question": inputs["question"],
            }

        rag_chain = (
            format_input
            | prompt
            | llm
            | StrOutputParser()
        )

    else:
        print("使用 Chroma 向量数据库 + Embedding 模型\n")

        # 准备文档
        from langchain_core.documents import Document

        documents = [
            Document(
                page_content=item["content"],
                metadata={"title": item["title"]},
            )
            for item in KNOWLEDGE_BASE
        ]

        # 创建向量数据库
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="langchain_tutorial",
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 构建 RAG 链
        template = """基于以下上下文回答问题。如果上下文中没有相关信息，请如实说明。

上下文:
{context}

问题: {question}

请用中文回答:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(
                f"[{d.metadata.get('title', '未知')}]\n{d.page_content}"
                for d in docs
            )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    # 测试问答
    questions = [
        "LangChain 是什么？有哪些核心组件？",
        "RAG 技术的工作原理是什么？",
        "向量数据库有哪些常见选择？",
        "文本切分有哪些策略？",
    ]

    for question in questions:
        print(f"❓ 问题: {question}")
        print("-" * 40)
        try:
            answer = rag_chain.invoke({"question": question})
            print(f"💬 回答: {answer}")
        except Exception as e:
            print(f"❌ 错误: {e}")
        print()


def main():
    print("\n" + "=" * 60)
    print("  模块 3: RAG（检索增强生成）")
    print("=" * 60 + "\n")

    demo_text_splitter()       # 示例 1: 文本切分
    embeddings = demo_embedding()  # 示例 2: 文本向量化
    demo_rag_concept()         # 示例 3: RAG 工作原理
    demo_full_rag(embeddings)  # 示例 4: 完整 RAG 系统

    print("=" * 60)
    print("  模块 3 学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
