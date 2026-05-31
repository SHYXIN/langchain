#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 2: Agents（智能体）
学习目标：
1. 理解 Agent 的概念和工作原理
2. 学会创建和使用工具（Tools）
3. 构建一个能自主决策的智能体
"""

import os
import sys
import io
import math
import json
from datetime import datetime

# 修复 Windows 终端编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# ============================================================
# 核心概念 1: Tools（工具）
# ============================================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。支持加减乘除、幂运算、开方等。
    示例: "2 + 3", "sqrt(16)", "2 ** 8"
    """
    try:
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息（模拟数据）"""
    weather_data = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 30},
        "上海": {"temp": 25, "condition": "多云", "humidity": 60},
        "广州": {"temp": 28, "condition": "小雨", "humidity": 80},
        "深圳": {"temp": 27, "condition": "阴", "humidity": 75},
        "杭州": {"temp": 23, "condition": "晴", "humidity": 45},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "未知", "humidity": 50})
    return json.dumps({
        "城市": city,
        "温度": f"{data['temp']}°C",
        "天气": data["condition"],
        "湿度": f"{data['humidity']}%"
    }, ensure_ascii=False)


# ============================================================
# 核心概念 2: 创建 Agent（LangChain v1 新 API）
# ============================================================

def create_demo_agent():
    """创建 Agent"""
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
    tools = [calculator, get_current_time, get_weather]
    return llm, tools


def demo_tools_standalone():
    """示例 1: 直接调用工具"""
    print("=" * 60)
    print("示例 1: 直接调用工具（不通过 LLM）")
    print("=" * 60)
    print(f"  calculator('2 + 3 * 4')  => {calculator.invoke('2 + 3 * 4')}")
    print(f"  calculator('sqrt(144)')   => {calculator.invoke('sqrt(144)')}")
    print(f"  calculator('2 ** 10')     => {calculator.invoke('2 ** 10')}")
    print(f"  get_current_time()        => {get_current_time.invoke({})}")
    print(f"  get_weather('北京')       => {get_weather.invoke('北京')}")
    print(f"  get_weather('上海')       => {get_weather.invoke('上海')}")
    print()


def demo_agent_concept():
    """示例 2: 展示 Agent 工作原理"""
    print("=" * 60)
    print("示例 2: Agent 工作原理")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────┐
│              Agent 执行循环（ReAct 模式）              │
│                                                     │
│  用户输入 ──→ [思考 Thought]                         │
│                  │                                  │
│                  ▼                                  │
│           [决定行动 Action]                          │
│                  │                                  │
│                  ▼                                  │
│           [执行工具 Tool]                            │
│                  │                                  │
│                  ▼                                  │
│           [观察结果 Observation]                     │
│                  │                                  │
│                  ▼                                  │
│           是否完成？──否──→ 继续思考                  │
│                  │                                  │
│                 是                                  │
│                  ▼                                  │
│           输出最终答案                                │
└─────────────────────────────────────────────────────┘

实际执行示例:
  用户: "北京天气怎么样？"
  ┌─ Thought: 需要查询北京天气
  │  Action: get_weather
  │  Input:  {"city": "北京"}
  │  Observation: {"城市":"北京","温度":"22°C","天气":"晴","湿度":"30%"}
  └─ Final Answer: 北京今天天气晴朗，温度 22°C，湿度 30%
    """)


def demo_agent_with_llm():
    """示例 3: 真实 Agent 调用（使用 LLM）"""
    print("=" * 60)
    print("示例 3: 真实 Agent 调用（使用 LLM）")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  未设置 OPENAI_API_KEY，跳过\n")
        return

    try:
        from langchain.agents import create_agent
    except ImportError:
        try:
            from langgraph.prebuilt import create_react_agent as create_agent
        except ImportError:
            print("⚠️  无法导入 create_agent，跳过\n")
            return

    llm, tools = create_demo_agent()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个有用的助手，可以使用工具来帮助用户解决问题。请用中文回答。",
    )

    questions = [
        "现在几点了？",
        "北京今天天气怎么样？",
        "帮我计算 2 的 10 次方是多少？",
    ]

    for question in questions:
        print(f"\n❓ 问题: {question}")
        print("-" * 40)
        try:
            result = agent.invoke({"messages": [("human", question)]})
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                answer = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                print(f"💬 回答: {answer}")
        except Exception as e:
            print(f"❌ 错误: {e}")
    print()


def main():
    print("\n" + "=" * 60)
    print("  模块 2: Agents（智能体）")
    print("=" * 60 + "\n")

    demo_tools_standalone()     # 示例 1
    demo_agent_concept()        # 示例 2
    demo_agent_with_llm()       # 示例 3

    print("=" * 60)
    print("  模块 2 学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
