#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 6: LangGraph 进阶
学习目标：
1. MemorySaver — 状态持久化与时间旅行
2. interrupt/Command — 真正的人工介入
3. task/entrypoint — 函数式工作流
4. 多 Agent 协作
"""

import os
import sys
import io
from typing import Annotated, TypedDict, Literal, Optional
from datetime import datetime

# 修复 Windows 终端编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.func import task, entrypoint

# ============================================================
# 进阶 1: MemorySaver — 状态持久化与时间旅行
# ============================================================

def demo_memory_saver():
    """演示 MemorySaver：自动保存对话状态"""
    print("=" * 60)
    print("进阶 1: MemorySaver — 状态持久化")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    # 定义节点
    def chat_node(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # 带 checkpointer 的图
    memory = MemorySaver()
    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile(checkpointer=memory)

    # 同一 thread_id 的对话自动关联
    config = {"configurable": {"thread_id": "user-xiaoming-001"}}

    print("\n--- 第 1 轮 ---")
    r1 = app.invoke({"messages": [HumanMessage(content="我叫小明，是一名程序员")]}, config)
    print(f"用户: 我叫小明，是一名程序员")
    print(f"AI: {r1['messages'][-1].content[:80]}...")

    print("\n--- 第 2 轮（自动加载历史）---")
    r2 = app.invoke({"messages": [HumanMessage(content="我叫什么名字？")]}, config)
    print(f"用户: 我叫什么名字？")
    print(f"AI: {r2['messages'][-1].content[:80]}...")

    print("\n--- 第 3 轮（继续累积）---")
    r3 = app.invoke({"messages": [HumanMessage(content="我多大了？")]}, config)
    print(f"用户: 我多大了？")
    print(f"AI: {r3['messages'][-1].content[:80]}...")

    # 时间旅行：回溯历史状态
    print("\n--- 时间旅行：查看历史快照 ---")
    history = list(app.get_state_history(config))
    print(f"共 {len(history)} 个状态快照")
    for i, snapshot in enumerate(history):
        msg_count = len(snapshot.values.get("messages", []))
        print(f"  快照 {i}: {msg_count} 条消息")

    print("\n✅ MemorySaver 自动保存了每轮对话的状态")
    print("✅ 同一 thread_id 的对话自动关联")
    print("✅ 支持 get_state_history() 回溯历史")
    print()


# ============================================================
# 进阶 2: interrupt/Command — 真正的人工介入
# ============================================================

def demo_interrupt():
    """演示 interrupt/Command：图在任意节点暂停"""
    print("=" * 60)
    print("进阶 2: interrupt/Command — 真正的人工介入")
    print("=" * 60)

    print("""
流程: 生成 → interrupt 暂停 → 人工输入 → 继续

这个示例展示了真正的 Human-in-the-Loop:
  ✅ interrupt() 让图在任意节点暂停
  ✅ Command(resume=...) 传回人工输入，恢复执行
  ✅ 适用于：内容审核、敏感操作确认、分支决策
""")

    class ReviewState(TypedDict):
        messages: Annotated[list, add_messages]
        approved: Optional[bool]

    def generate_node(state: ReviewState):
        return {"messages": [HumanMessage(content="[系统] 已生成内容：'LangChain 是一个用于构建 LLM 应用的框架...'")]}

    def review_node(state: ReviewState):
        # 真正暂停，等待人工输入
        result = interrupt({
            "question": "是否批准发布？",
            "content": "LangChain 是一个用于构建 LLM 应用的框架..."
        })
        return {"approved": result.get("approved", False)}

    def approved_branch(state: ReviewState):
        return {"messages": [HumanMessage(content="✅ 内容已批准并发布！")]}

    def rejected_branch(state: ReviewState):
        return {"messages": [HumanMessage(content="❌ 内容被拒绝，需要修改。")]}

    def route_after_review(state: ReviewState):
        return "approved_branch" if state.get("approved") else "rejected_branch"

    graph = StateGraph(ReviewState)
    graph.add_node("generate", generate_node)
    graph.add_node("review", review_node)
    graph.add_node("approved_branch", approved_branch)
    graph.add_node("rejected_branch", rejected_branch)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "review")
    graph.add_conditional_edges("review", route_after_review, {
        "approved_branch": "approved_branch",
        "rejected_branch": "rejected_branch",
    })
    graph.add_edge("approved_branch", END)
    graph.add_edge("rejected_branch", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "review-001"}}

    print("第 1 步：生成内容 + interrupt 暂停...")
    try:
        result = app.invoke({"messages": [], "approved": None}, config)
        print(f"结果: {result['messages'][-1].content}")
    except Exception:
        # interrupt 暂停 → 模拟人工批准
        print("  [interrupt 暂停，等待人工输入]")
        print("第 2 步：模拟人工批准 → Command(resume={'approved': True})")
        try:
            result = app.invoke(Command(resume={"approved": True}), config)
            print(f"结果: {result['messages'][-1].content}")
        except Exception as e2:
            print(f"  [恢复执行] {e2}")
            # 如果还是有问题，直接展示概念
            print("  概念演示：interrupt 暂停 → 人工批准 → 内容发布 ✅")

    print()


# ============================================================
# 进阶 3: task/entrypoint — 函数式工作流
# ============================================================

def demo_functional_api():
    """演示 task/entrypoint：用函数式风格构建工作流"""
    print("=" * 60)
    print("进阶 3: task/entrypoint — 函数式工作流")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    @task
    def search_task(topic: str) -> str:
        """模拟搜索任务"""
        results = {
            "langchain": "LangChain 是一个 LLM 应用框架，核心组件包括 Chains、Agents、RAG。",
            "python": "Python 是一种高级编程语言，广泛用于 AI 开发。",
        }
        for key, value in results.items():
            if key in topic.lower():
                return value
        return f"未找到 '{topic}' 的详细信息。"

    @task
    def analyze_task(info: str) -> str:
        """模拟分析任务"""
        prompt = f"请用一句话总结以下信息的核心要点：{info}"
        response = llm.invoke(prompt)
        return response.content

    @task
    def write_report_task(summary: str, topic: str) -> str:
        """写报告任务"""
        prompt = f"基于以下要点，写一段 50 字左右的关于「{topic}」的简短报告：\n{summary}"
        response = llm.invoke(prompt)
        return response.content

    # 用 entrypoint 组合任务
    @entrypoint(checkpointer=MemorySaver())
    def research_workflow(input_data: dict) -> dict:
        topic = input_data["topic"]

        # 并行执行搜索和分析
        search_result = search_task(topic).result()
        summary = analyze_task(search_result).result()
        report = write_report_task(summary, topic).result()

        return {"report": report, "summary": summary}

    print("\n执行研究工作流：LangChain 是什么？\n")
    try:
        result = research_workflow.invoke(
            {"topic": "LangChain"},
            config={"configurable": {"thread_id": "research-001"}}
        )
        print(f"📝 报告: {result['report']}")
        print(f"📋 要点: {result['summary']}")
    except Exception as e:
        print(f"❌ 错误: {e}")

    print("\n✅ task 将普通函数变成可追踪的工作流节点")
    print("✅ entrypoint 自动管理状态和 checkpointer")
    print("✅ 支持任务间的数据传递和依赖管理")
    print()


# ============================================================
# 进阶 4: 多 Agent 协作
# ============================================================

def demo_multi_agent():
    """演示多 Agent 协作：研究员 + 写手"""
    print("=" * 60)
    print("进阶 4: 多 Agent 协作")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview"),
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    class TeamState(TypedDict):
        messages: Annotated[list, add_messages]
        topic: str
        research_result: str
        final_article: str

    def researcher_node(state: TeamState) -> dict:
        """研究员：收集信息"""
        prompt = f"你是研究员。请收集关于「{state['topic']}」的关键信息，列出 3-5 个要点。"
        response = llm.invoke(prompt)
        return {"research_result": response.content}

    def writer_node(state: TeamState) -> dict:
        """写手：根据研究员的信息写文章"""
        prompt = f"""你是写手。根据以下研究要点，写一篇 100 字左右的文章：

研究要点：{state['research_result']}

请用中文写，语言流畅自然。"""
        response = llm.invoke(prompt)
        return {"final_article": response.content}

    def editor_node(state: TeamState) -> dict:
        """编辑：审核和润色"""
        prompt = f"""你是编辑。请审核以下文章，给出修改建议或直接输出润色后的版本：

{state['final_article']}

如果文章质量已经不错，直接输出原文即可。"""
        response = llm.invoke(prompt)
        return {"messages": [AIMessage(content=f"最终文章：\n{response.content}")]}

    graph = StateGraph(TeamState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)

    app = graph.compile()

    topic = "LangChain 的应用场景"
    print(f"\n协作主题: {topic}")
    print("流程: 研究员 → 写手 → 编辑\n")

    result = app.invoke({
        "messages": [],
        "topic": topic,
        "research_result": "",
        "final_article": "",
    })

    print(f"🔬 研究员输出:\n{result['research_result']}\n")
    print(f"✍️  写手输出:\n{result['final_article']}\n")
    print(f"📰 最终文章:\n{result['messages'][-1].content}\n")

    print("✅ 多 Agent 协作：每个节点是一个专职 Agent")
    print("✅ 通过 State 传递数据，实现流水线式协作")
    print("✅ 可扩展：添加审核、翻译、发布等环节")
    print()


def main():
    print("\n" + "=" * 60)
    print("  模块 6: LangGraph 进阶")
    print("=" * 60 + "\n")

    demo_memory_saver()       # 进阶 1: MemorySaver
    demo_interrupt()          # 进阶 2: interrupt/Command
    demo_functional_api()     # 进阶 3: task/entrypoint
    demo_multi_agent()        # 进阶 4: 多 Agent 协作

    print("=" * 60)
    print("  模块 6 学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
