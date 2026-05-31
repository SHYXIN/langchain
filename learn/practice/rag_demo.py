#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能客服 RAG 演示系统
基于 Bitext 客服数据集（中文版）

用法:
  uv run python rag_demo.py              # 交互式问答
  uv run python rag_demo.py --test       # 自动测试模式
  uv run python rag_demo.py --stats      # 查看数据统计
"""

import os
import sys
import io
import json
import glob
import argparse
from pathlib import Path
from typing import Optional

# ── 编码修复 ──────────────────────────────────────────
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

# ── 配置 ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "bitext_cn"
EMBEDDING_MODEL_PATH = Path(__file__).parent / "models" / "BAAI" / "bge-small-zh-v1.5"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# ── 加载数据 ──────────────────────────────────────────
def load_all_data() -> list[dict]:
    """加载所有翻译后的客服数据"""
    all_data = []
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        print(f"❌ 未找到数据文件，请先运行翻译脚本")
        sys.exit(1)

    for fpath in files:
        category = fpath.stem
        with open(fpath, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            item["category"] = category
        all_data.extend(items)
        print(f"  ✅ {category}: {len(items)} 条")

    return all_data


# ── 关键词检索（无需 embedding 模型） ─────────────────
class KeywordRetriever:
    """基于关键词重叠的简单检索器"""

    def __init__(self, data: list[dict]):
        self.data = data

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_words = set(query.lower().split())
        scored = []
        for item in self.data:
            # 在 instruction + response 中匹配
            text = f"{item.get('instruction', '')} {item.get('response', '')}"
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]


# ── 向量检索（Chroma + Embedding） ────────────────────
class VectorRetriever:
    """基于向量的语义检索器"""

    def __init__(self, data: list[dict], collection_name: str = "customer_service"):
        self.data = data
        self.collection_name = collection_name
        self._init_vectorstore()

    def _init_vectorstore(self):
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document

        # 检查是否已有持久化的数据库
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            print("  📂 加载已有向量数据库...")
            self.vectorstore = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self._get_embedding(),
                collection_name=self.collection_name,
            )
        else:
            print("  🔨 构建向量数据库（首次运行需要几分钟）...")
            documents = []
            for i, item in enumerate(self.data):
                doc = Document(
                    page_content=f"问题: {item['instruction']}\n回答: {item['response']}",
                    metadata={
                        "id": item.get("id", str(i)),
                        "category": item.get("category", ""),
                        "intent": item.get("intent", ""),
                        "instruction": item["instruction"],
                        "response": item["response"],
                    },
                )
                documents.append(doc)

            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self._get_embedding(),
                persist_directory=str(CHROMA_DIR),
                collection_name=self.collection_name,
            )
            print(f"  ✅ 向量数据库构建完成，共 {len(documents)} 条文档")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def _get_embedding(self):
        """获取 embedding 模型，优先本地 BGE，回退到 OpenAI"""
        if EMBEDDING_MODEL_PATH.exists():
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=str(EMBEDDING_MODEL_PATH),
                model_kwargs={"device": "cpu"},
            )
        else:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        from langchain_core.documents import Document
        docs = self.retriever.invoke(query)
        results = []
        for doc in docs:
            meta = doc.metadata
            results.append({
                "instruction": meta.get("instruction", ""),
                "response": meta.get("response", ""),
                "category": meta.get("category", ""),
                "intent": meta.get("intent", ""),
                "score": getattr(doc, "score", 0),
            })
        return results


# ── RAG 引擎 ──────────────────────────────────────────
class CustomerServiceRAG:
    """智能客服 RAG 系统"""

    SYSTEM_PROMPT = """你是一个专业的智能客服助手。请根据以下知识库内容回答用户问题。

规则：
1. 如果知识库中有相关信息，请基于知识库内容回答，保持专业和友好
2. 如果知识库中没有相关信息，请礼貌地告知用户并建议联系人工客服
3. 回答要简洁明了，不要过度冗长
4. 使用中文回答

知识库内容：
{context}"""

    def __init__(self, use_vector: bool = True):
        print("=" * 60)
        print("  🤖 智能客服 RAG 系统")
        print("=" * 60)

        print("\n📥 加载客服知识库...")
        self.data = load_all_data()
        print(f"  总计: {len(self.data)} 条知识\n")

        # 选择检索方式
        if use_vector:
            try:
                print("🔍 初始化向量检索...")
                self.retriever = VectorRetriever(self.data)
                self.mode = "vector"
            except Exception as e:
                print(f"  ⚠️ 向量检索初始化失败: {e}")
                print("  🔄 回退到关键词检索...")
                self.retriever = KeywordRetriever(self.data)
                self.mode = "keyword"
        else:
            self.retriever = KeywordRetriever(self.data)
            self.mode = "keyword"

        # 初始化 LLM
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm_kwargs = {"model": model_name, "temperature": 0.3, "api_key": api_key}
        if base_url:
            llm_kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**llm_kwargs)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "用户问题：{question}"),
        ])
        self.parser = StrOutputParser()

        print(f"\n  ✅ 就绪！检索模式: {self.mode}")
        print(f"  ✅ LLM: {model_name}")
        print()

    def _format_context(self, items: list[dict]) -> str:
        if not items:
            return "（未找到相关知识库内容）"
        parts = []
        for i, item in enumerate(items, 1):
            part = f"[{i}] 问题: {item['instruction']}\n    回答: {item['response']}"
            if item.get("category"):
                part = f"    分类: {item['category']}\n" + part
            parts.append(part)
        return "\n\n".join(parts)

    def ask(self, question: str) -> str:
        """提问并获取回答"""
        # 检索
        retrieved = self.retriever.retrieve(question, k=3)
        context = self._format_context(retrieved)

        # 生成回答
        chain = self.prompt | self.llm | self.parser
        response = chain.invoke({"context": context, "question": question})
        return response, retrieved

    def interactive(self):
        """交互式问答"""
        print("─" * 60)
        print("  输入问题进行咨询，输入 'quit' 退出")
        print("  输入 'debug' 切换调试模式")
        print("─" * 60)

        debug = False
        while True:
            try:
                question = input("\n👤 用户: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not question:
                continue
            if question.lower() == "quit":
                print("再见！")
                break
            if question.lower() == "debug":
                debug = not debug
                print(f"调试模式: {'开启' if debug else '关闭'}")
                continue

            response, retrieved = self.ask(question)

            if debug:
                print(f"\n🔍 检索到 {len(retrieved)} 条相关文档:")
                for i, item in enumerate(retrieved, 1):
                    print(f"  [{i}] [{item.get('category', '?')}] {item['instruction'][:60]}...")

            print(f"\n🤖 客服: {response}")

    def run_tests(self):
        """自动测试"""
        test_questions = [
            "怎么取消订单？",
            "如何申请退款？",
            "忘记密码怎么办？",
            "修改收货地址",
            "有哪些支付方式？",
            "订单什么时候能到？",
            "怎么联系人工客服？",
            "如何开发票？",
            "怎么注销账户？",
            "投诉你们的服务",
        ]

        print("─" * 60)
        print("  🧪 自动测试模式")
        print("─" * 60)

        for q in test_questions:
            response, retrieved = self.ask(q)
            print(f"\n👤 用户: {q}")
            if retrieved:
                cats = [item.get("category", "?") for item in retrieved]
                print(f"🔍 检索: {len(retrieved)} 条 | 分类: {', '.join(cats)}")
            else:
                print("🔍 检索: 未找到相关内容")
            print(f"🤖 客服: {response[:200]}...")
            print("─" * 40)


# ── 统计信息 ──────────────────────────────────────────
def show_stats():
    """显示数据统计"""
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        print("❌ 未找到数据文件")
        return

    print("=" * 60)
    print("  📊 数据统计")
    print("=" * 60)

    total = 0
    for fpath in files:
        category = fpath.stem
        with open(fpath, "r", encoding="utf-8") as f:
            items = json.load(f)
        count = len(items)
        total += count

        # 统计 intent 分布
        intents = {}
        for item in items:
            intent = item.get("intent", "unknown")
            intents[intent] = intents.get(intent, 0) + 1

        print(f"\n  📁 {category}: {count} 条")
        for intent, cnt in sorted(intents.items(), key=lambda x: -x[1])[:5]:
            print(f"     - {intent}: {cnt}")

    print(f"\n  📊 总计: {total} 条")
    print(f"  📊 分类: {len(files)} 个")


# ── 主入口 ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能客服 RAG 演示")
    parser.add_argument("--test", action="store_true", help="自动测试模式")
    parser.add_argument("--stats", action="store_true", help="查看数据统计")
    parser.add_argument("--no-vector", action="store_true", help="使用关键词检索（不用向量）")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        rag = CustomerServiceRAG(use_vector=not args.no_vector)
        if args.test:
            rag.run_tests()
        else:
            rag.interactive()
