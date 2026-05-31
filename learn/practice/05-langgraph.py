#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 5: LangGraph 工作流
学习目标：
1. 理解 LangGraph 的核心概念（State、Node、Edge）
2. 学会构建简单的状态图
3. 构建带条件分支和循环的工作流
4. 构建一个完整的 AI 工作流应用
"""

import os
import sys
import io
import json
import operator
from typing import Annotated, TypedDict, Literal
from datetime import datetime

# 修复 Windows 终端编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 核心概念 1: State（状态）
# ============================================================

def demo_state_concept():
    """演示 State 的概念"""
    print("=" * 60)
    print("示例 1: State（状态）概念")
    print("=" * 60)
    print("""
LangGraph 的核心是 状态图（StateGraph）:

  ┌──────────────────────────────────────────────────┐
  │                  State（状态）                     │
  │  图中的"全局数据"，所有节点共享和更新              │
  │  用 TypedDict 定义结构                            │
  └──────────────────────────────────────────────────┘

  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │  Node A  │────→│  Node B  │────→│  Node C  │
  │ (处理)   │     │ (处理)   │     │ (处理)   │
  └─────────┘     └─────────┘     └─────────┘
       │                │                │
       ▼                ▼                ▼
  ┌──────────────────────────────────────────────────┐
  │              State 被逐步更新                      │
  │  {messages: [...], step: "A"}                    │
  │  {messages: [...], step: "B"}                    │
  │  {messages: [...], step: "C"}                    │
  └──────────────────────────────────────────────────┘

关键概念:
  - State:   图中流动的数据（用 TypedDict 定义）
  - Node:    处理状态的函数（接收 State，返回更新）
  - Edge:    节点之间的连接（普通边 / 条件边）
  - START:   图的入口
  - END:     图的出口
    """)
    print()


# ============================================================
# 核心概念 2: 最简单的图
# ============================================================

class SimpleState(TypedDict):
    """最简单的状态：包含消息列表"""
    messages: Annotated[list, add_messages]
    step: str


def node_a(state: SimpleState) -> dict:
    """节点 A：添加一条消息"""
    return {
        "messages": [HumanMessage(content="节点 A 处理完成")],
        "step": "A",
    }


def node_b(state: SimpleState) -> dict:
    """节点 B：添加一条消息"""
    return {
        "messages": [HumanMessage(content="节点 B 处理完成")],
        "step": "B",
    }


def demo_simple_graph():
    """演示最简单的图"""
    print("=" * 60)
    print("示例 2: 最简单的图（START → A → B → END）")
    print("=" * 60)

    # 创建图
    graph = StateGraph(SimpleState)

    # 添加节点
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)

    # 添加边
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)

    # 编译图
    app = graph.compile()

    # 执行
    result = app.invoke({"messages": [], "step": ""})

    print("执行结果:")
    for msg in result["messages"]:
        print(f"  → {msg.content}")
    print(f"  最终 step: {result['step']}")
    print()


# ============================================================
# 核心概念 3: 条件分支
# ============================================================

class BranchState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str


def start_node(state: BranchState) -> dict:
    """起始节点"""
    return {"messages": [HumanMessage(content="开始处理...")]}


def positive_node(state: BranchState) -> dict:
    """正面处理节点"""
    return {
        "messages": [HumanMessage(content="😊 检测到正面情绪，执行正面流程")],
        "route": "positive",
    }


def negative_node(state: BranchState) -> dict:
    """负面处理节点"""
    return {
        "messages": [HumanMessage(content="😢 检测到负面情绪，执行关怀流程")],
        "route": "negative",
    }


def decide_route(state: BranchState) -> Literal["positive", "negative"]:
    """条件路由函数：根据消息内容决定走向"""
    last_msg = state["messages"][-1].content if state["messages"] else ""
    positive_words = ["好", "棒", "开心", "喜欢", "优秀", "good", "great", "happy"]
    if any(word in last_msg.lower() for word in positive_words):
        return "positive"
    return "negative"


def demo_conditional_graph():
    """演示条件分支"""
    print("=" * 60)
    print("示例 3: 条件分支（根据输入选择不同路径）")
    print("=" * 60)

    graph = StateGraph(BranchState)
    graph.add_node("start", start_node)
    graph.add_node("positive", positive_node)
    graph.add_node("negative", negative_node)

    graph.add_edge(START, "start")
    graph.add_conditional_edges(
        "start",
        decide_route,
        {
            "positive": "positive",
            "negative": "negative",
        },
    )
    graph.add_edge("positive", END)
    graph.add_edge("negative", END)

    app = graph.compile()

    # 测试正面输入
    print("输入: '今天天气真好！'")
    result = app.invoke({"messages": [HumanMessage(content="今天天气真好！")], "route": ""})
    print(f"  路由结果: {result['route']}")
    print(f"  消息: {result['messages'][-1].content}")

    # 测试负面输入
    print("\n输入: '今天心情很差'")
    result = app.invoke({"messages": [HumanMessage(content="今天心情很差")], "route": ""})
    print(f"  路由结果: {result['route']}")
    print(f"  消息: {result['messages'][-1].content}")
    print()


# ============================================================
# 核心概念 4: 循环（Cycle）
# ============================================================

class LoopState(TypedDict):
    messages: Annotated[list, add_messages]
    counter: int
    max_iterations: int


def process_node(state: LoopState) -> dict:
    """处理节点：每次执行计数 +1"""
    counter = state.get("counter", 0) + 1
    return {
        "messages": [HumanMessage(content=f"第 {counter} 次处理")],
        "counter": counter,
    }


def should_continue(state: LoopState) -> Literal["continue", "stop"]:
    """决定是否继续循环"""
    if state["counter"] >= state["max_iterations"]:
        return "stop"
    return "continue"


def demo_loop_graph():
    """演示循环"""
    print("=" * 60)
    print("示例 4: 循环（处理 3 次后停止）")
    print("=" * 60)

    graph = StateGraph(LoopState)
    graph.add_node("process", process_node)
    graph.add_edge(START, "process")
    graph.add_conditional_edges(
        "process",
        should_continue,
        {
            "continue": "process",  # 循环回自身
            "stop": END,
        },
    )

    app = graph.compile()

    result = app.invoke({
        "messages": [],
        "counter": 0,
        "max_iterations": 3,
    })

    print("执行过程:")
    for msg in result["messages"]:
        print(f"  → {msg.content}")
    print(f"  总执行次数: {result['counter']}")
    print()


# ============================================================
# 核心概念 5: 完整的 AI Agent 工作流
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def search_web(query: str) -> str:
    """搜索网络获取信息（模拟）"""
    results = {
        "langchain": "LangChain 是一个 LLM 应用框架，支持 Chains、Agents、RAG 等功能。",
        "python": "Python 是一种高级编程语言，广泛用于 AI、Web 开发等领域。",
        "北京天气": "北京今天晴，温度 22°C，湿度 30%。",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"未找到 '{query}' 的相关信息。"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def create_ai_agent():
    """创建 AI Agent"""
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
    tools = [search_web, calculate]
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools, tools


def agent_node(state: AgentState) -> dict:
    """Agent 节点：调用 LLM"""
    llm_with_tools, _ = create_ai_agent()
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def demo_ai_agent_workflow():
    """演示完整的 AI Agent 工作流"""
    print("=" * 60)
    print("示例 5: 完整的 AI Agent 工作流")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  未设置 OPENAI_API_KEY，跳过\n")
        return

    llm_with_tools, tools = create_ai_agent()

    # 构建图
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,  # 内置的工具路由函数
        {
            "tools": "tools",
            END: END,
        },
    )
    graph.add_edge("tools", "agent")  # 工具执行后回到 agent

    app = graph.compile()

    questions = [
        "LangChain 是什么？",
        "帮我计算 2 的 10 次方",
    ]

    for question in questions:
        print(f"\n❓ 问题: {question}")
        print("-" * 40)
        try:
            result = app.invoke({
                "messages": [HumanMessage(content=question)]
            })
            for msg in result["messages"]:
                if hasattr(msg, 'content') and msg.content:
                    print(f"  [{type(msg).__name__}] {msg.content[:100]}")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    print()


# ============================================================
# 核心概念 6: 人工干预（Human-in-the-Loop）
# ============================================================

class HumanLoopState(TypedDict):
    messages: Annotated[list, add_messages]
    approved: bool


def generate_content(state: HumanLoopState) -> dict:
    """生成内容"""
    return {
        "messages": [HumanMessage(content="生成的内容: 这是一篇关于 AI 的文章...")],
    }


def human_review(state: HumanLoopState) -> dict:
    """人工审核（模拟）"""
    # 模拟人工审核通过
    return {"approved": True}


def publish_content(state: HumanLoopState) -> dict:
    """发布内容"""
    return {
        "messages": [HumanMessage(content="✅ 内容已发布！")],
    }


def revise_content(state: HumanLoopState) -> dict:
    """修改内容"""
    return {
        "messages": [HumanMessage(content="🔄 内容已修改，重新提交审核")],
    }


def review_decision(state: HumanLoopState) -> Literal["publish", "revise"]:
    """审核决策"""
    if state.get("approved"):
        return "publish"
    return "revise"


def demo_human_in_loop():
    """演示 Human-in-the-Loop 工作流"""
    print("=" * 60)
    print("示例 6: Human-in-the-Loop 工作流")
    print("=" * 60)
    print("""
流程:
  生成内容 → 人工审核 → 通过 → 发布
                      → 不通过 → 修改 → 重新审核

这个示例展示了 LangGraph 的核心优势:
  ✅ 可以在任意节点暂停，等待人工输入
  ✅ 支持条件分支和循环
  ✅ 状态在整个工作流中持久化
    """)

    graph = StateGraph(HumanLoopState)
    graph.add_node("generate", generate_content)
    graph.add_node("review", human_review)
    graph.add_node("publish", publish_content)
    graph.add_node("revise", revise_content)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "review")
    graph.add_conditional_edges(
        "review",
        review_decision,
        {
            "publish": "publish",
            "revise": "revise",
        },
    )
    graph.add_edge("revise", "review")  # 修改后重新审核
    graph.add_edge("publish", END)

    app = graph.compile()

    result = app.invoke({"messages": [], "approved": False})

    print("执行结果:")
    for msg in result["messages"]:
        print(f"  → {msg.content}")
    print()


def main():
    print("\n" + "=" * 60)
    print("  模块 5: LangGraph 工作流")
    print("=" * 60 + "\n")

    demo_state_concept()          # 示例 1: State 概念
    demo_simple_graph()           # 示例 2: 最简单的图
    demo_conditional_graph()      # 示例 3: 条件分支
    demo_loop_graph()             # 示例 4: 循环
    demo_ai_agent_workflow()      # 示例 5: AI Agent 工作流
    demo_human_in_loop()          # 示例 6: Human-in-the-Loop

    print("=" * 60)
    print("  模块 5 学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
