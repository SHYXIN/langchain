#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 4: Memory（记忆管理）
学习目标：
1. 理解为什么需要 Memory
2. 学会手动管理对话历史
3. 实现滑动窗口记忆
4. 实现摘要记忆
5. 使用 LangGraph Checkpoint 自动管理记忆
"""

import os
import sys
import io
import math
from datetime import datetime
from collections import deque

# 修复 Windows 终端编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 核心概念 1: 无 Memory vs 有 Memory
# ============================================================

def demo_no_memory():
    """演示：没有 Memory 的对话"""
    print("=" * 60)
    print("示例 1: 没有 Memory 的对话")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")

    llm = ChatOpenAI(
        model=model_name, temperature=0,
        api_key=api_key, base_url=base_url,
    )

    # 第一轮对话
    print("--- 第 1 轮 ---")
    response = llm.invoke([HumanMessage(content="我叫小明，今年 25 岁")])
    print(f"用户: 我叫小明，今年 25 岁")
    print(f"AI: {response.content}")
    print()

    # 第二轮对话 — 没有传入历史消息，AI 不记得之前的话
    print("--- 第 2 轮（没有历史）---")
    response = llm.invoke([HumanMessage(content="我叫什么名字？")])
    print(f"用户: 我叫什么名字？")
    print(f"AI: {response.content}")
    print("⚠️  AI 不记得你的名字，因为没有传入对话历史！")
    print()


def demo_with_memory():
    """演示：手动管理 Memory"""
    print("=" * 60)
    print("示例 2: 手动管理 Memory")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")

    llm = ChatOpenAI(
        model=model_name, temperature=0,
        api_key=api_key, base_url=base_url,
    )

    # 手动维护对话历史
    conversation_history = []

    # 第 1 轮
    print("--- 第 1 轮 ---")
    user_msg = "我叫小明，今年 25 岁，喜欢编程"
    print(f"用户: {user_msg}")
    conversation_history.append(HumanMessage(content=user_msg))

    response = llm.invoke(conversation_history)
    print(f"AI: {response.content}")
    conversation_history.append(response)  # 保存 AI 回复
    print()

    # 第 2 轮
    print("--- 第 2 轮（带历史）---")
    user_msg = "我叫什么名字？"
    print(f"用户: {user_msg}")
    conversation_history.append(HumanMessage(content=user_msg))

    response = llm.invoke(conversation_history)
    print(f"AI: {response.content}")
    conversation_history.append(response)
    print("✅ AI 记住了你的名字！")
    print()

    # 第 3 轮
    print("--- 第 3 轮（带历史）---")
    user_msg = "我有什么爱好？"
    print(f"用户: {user_msg}")
    conversation_history.append(HumanMessage(content=user_msg))

    response = llm.invoke(conversation_history)
    print(f"AI: {response.content}")
    conversation_history.append(response)
    print("✅ AI 记住了你的爱好！")
    print()

    # 打印完整对话历史
    print("--- 完整对话历史 ---")
    for i, msg in enumerate(conversation_history):
        role = type(msg).__name__.replace("Message", "")
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  [{i+1}] {role}: {content}")
    print()


# ============================================================
# 核心概念 2: 滑动窗口 Memory
# ============================================================

class SlidingWindowMemory:
    """滑动窗口记忆：只保留最近 K 条消息"""

    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self.messages: list = []

    def add_message(self, message):
        """添加消息"""
        self.messages.append(message)

    def get_messages(self) -> list:
        """获取最近 K 条消息"""
        return self.messages[-self.window_size:]

    def get_buffer_string(self) -> str:
        """获取可读的对话历史"""
        lines = []
        for msg in self.get_messages():
            role = type(msg).__name__.replace("Message", "")
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self):
        """清空记忆"""
        self.messages = []


def demo_sliding_window_memory():
    """演示：滑动窗口记忆"""
    print("=" * 60)
    print("示例 3: 滑动窗口 Memory")
    print("=" * 60)

    memory = SlidingWindowMemory(window_size=4)

    # 模拟多轮对话
    conversations = [
        ("human", "你好！"),
        ("ai", "你好！有什么可以帮助你的吗？"),
        ("human", "我叫小明"),
        ("ai", "你好小明！很高兴认识你。"),
        ("human", "我喜欢编程"),
        ("ai", "编程很棒！你学的是什么语言？"),
        ("human", "Python"),
        ("ai", "Python 是很流行的语言！"),
        ("human", "LangChain 是什么？"),
        ("ai", "LangChain 是一个 LLM 框架..."),
    ]

    print(f"窗口大小: {memory.window_size}")
    print()

    for role, content in conversations:
        if role == "human":
            memory.add_message(HumanMessage(content=content))
        else:
            memory.add_message(AIMessage(content=content))

    print(f"总消息数: {len(memory.messages)}")
    print(f"窗口内消息数: {len(memory.get_messages())}")
    print()
    print("滑动窗口内的消息:")
    for msg in memory.get_messages():
        role = type(msg).__name__.replace("Message", "")
        print(f"  {role}: {msg.content}")
    print()
    print("✅ 滑动窗口自动丢弃了早期消息，保留最近的对话")
    print()


# ============================================================
# 核心概念 3: 摘要 Memory
# ============================================================

class SummaryMemory:
    """摘要记忆：将旧的对话总结为摘要"""

    def __init__(self, llm, summary_threshold: int = 6):
        self.llm = llm
        self.summary_threshold = summary_threshold
        self.messages: list = []
        self.summary: str = ""

    def add_message(self, message):
        self.messages.append(message)

    def _summarize(self, messages: list) -> str:
        """将消息列表总结为摘要"""
        if not messages:
            return ""

        # 构建总结提示
        conversation = "\n".join(
            f"{type(m).__name__.replace('Message', '')}: {m.content}"
            for m in messages
        )

        prompt = f"""请用一句话总结以下对话的关键信息（用户偏好、重要事实等）:

{conversation}

总结:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            return "（总结失败）"

    def get_messages(self) -> list:
        """获取消息：摘要 + 最近的消息"""
        if len(self.messages) <= self.summary_threshold:
            return self.messages

        # 需要总结的旧消息
        old_messages = self.messages[:-self.summary_threshold]
        recent_messages = self.messages[-self.summary_threshold:]

        # 更新摘要
        if old_messages:
            new_summary = self._summarize(old_messages)
            if self.summary:
                self.summary = f"{self.summary}\n{new_summary}"
            else:
                self.summary = new_summary

        # 返回：摘要消息 + 最近消息
        result = []
        if self.summary:
            result.append(SystemMessage(
                content=f"对话历史摘要：{self.summary}"
            ))
        result.extend(recent_messages)

        # 清空已总结的消息
        self.messages = recent_messages

        return result

    def clear(self):
        self.messages = []
        self.summary = ""


def demo_summary_memory():
    """演示：摘要记忆"""
    print("=" * 60)
    print("示例 4: 摘要 Memory")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")

    llm = ChatOpenAI(
        model=model_name, temperature=0,
        api_key=api_key, base_url=base_url,
    )

    memory = SummaryMemory(llm, summary_threshold=4)

    print(f"总结阈值: {memory.summary_threshold} 条消息")
    print()

    # 模拟多轮对话
    conversations = [
        ("human", "我叫小明，今年 25 岁"),
        ("ai", "你好小明！"),
        ("human", "我喜欢打篮球和编程"),
        ("ai", "很棒的爱好！"),
        ("human", "我在北京工作，是一名软件工程师"),
        ("ai", "北京是个很棒的城市！"),
        ("human", "我正在学习 LangChain"),
        ("ai", "LangChain 很有用！"),
    ]

    for i, (role, content) in enumerate(conversations):
        if role == "human":
            memory.add_message(HumanMessage(content=content))
        else:
            memory.add_message(AIMessage(content=content))

        total = len(memory.messages)
        print(f"  [{i+1}] {role}: {content}  (记忆: {total} 条)")

    print()
    print("获取消息（触发摘要）:")
    messages = memory.get_messages()
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  {role}: {content}")
    print()
    print("✅ 早期对话被压缩为摘要，保留了关键信息")
    print()


# ============================================================
# 核心概念 4: 完整的对话系统
# ============================================================

class ChatBot:
    """带记忆的对话机器人"""

    def __init__(self, memory_type: str = "sliding", window_size: int = 10):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")

        self.llm = ChatOpenAI(
            model=model_name, temperature=0.7,
            api_key=api_key, base_url=base_url,
        )

        if memory_type == "sliding":
            self.memory = SlidingWindowMemory(window_size=window_size)
        elif memory_type == "summary":
            self.memory = SummaryMemory(self.llm, summary_threshold=window_size)
        else:
            self.memory = SlidingWindowMemory(window_size=100)

        self.system_prompt = "你是一个友好的对话助手，可以用中文和用户交流。"

    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        # 添加用户消息到记忆
        self.memory.add_message(HumanMessage(content=user_input))

        # 获取带记忆的消息
        messages = self.memory.get_messages()

        # 添加系统提示
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=self.system_prompt))

        # 调用 LLM
        response = self.llm.invoke(messages)

        # 保存 AI 回复到记忆
        self.memory.add_message(response)

        return response.content

    def get_history(self) -> str:
        """获取对话历史"""
        if hasattr(self.memory, 'get_buffer_string'):
            return self.memory.get_buffer_string()
        return "\n".join(
            f"{type(m).__name__.replace('Message', '')}: {m.content}"
            for m in self.memory.messages
        )


def demo_chatbot():
    """演示：完整的对话机器人"""
    print("=" * 60)
    print("示例 5: 带记忆的对话机器人")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  未设置 OPENAI_API_KEY，跳过\n")
        return

    bot = ChatBot(memory_type="sliding", window_size=6)

    # 模拟对话
    conversations = [
        "你好！我叫小明",
        "我今年 25 岁，是一名程序员",
        "我最喜欢的编程语言是 Python",
        "我叫什么名字？",
        "我多大了？",
        "我是做什么工作的？",
    ]

    for user_input in conversations:
        print(f"👤 用户: {user_input}")
        response = bot.chat(user_input)
        print(f"🤖 AI: {response}")
        print()

    print("--- 对话历史 ---")
    print(bot.get_history())
    print()


def main():
    print("\n" + "=" * 60)
    print("  模块 4: Memory（记忆管理）")
    print("=" * 60 + "\n")

    api_key = os.getenv("OPENAI_API_KEY")

    demo_no_memory()               # 示例 1: 无 Memory
    demo_with_memory()             # 示例 2: 手动 Memory
    demo_sliding_window_memory()   # 示例 3: 滑动窗口
    demo_summary_memory()          # 示例 4: 摘要 Memory
    demo_chatbot()                 # 示例 5: 完整对话机器人

    print("=" * 60)
    print("  模块 4 学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
