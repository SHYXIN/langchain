#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Bitext 客服数据集中每个 category 取 100 条，翻译成中文，保存为 JSON。
使用 OpenAI 兼容 API 批量翻译。
"""

import os
import json
import asyncio
import sys
import io
from pathlib import Path
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 加载 .env（从仓库根目录或当前目录）
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from openai import AsyncOpenAI

# ── 配置 ──────────────────────────────────────────────
SAMPLE_PER_CATEGORY = 100
OUTPUT_DIR = Path(__file__).parent / "bitext_cn"
OUTPUT_DIR.mkdir(exist_ok=True)

# 10 个分类（按业务重要性排序）
CATEGORIES = [
    "ORDER",
    "REFUND",
    "ACCOUNT",
    "PAYMENT",
    "DELIVERY",
    "SHIPPING",
    "INVOICE",
    "CONTACT",
    "FEEDBACK",
    "SUBSCRIPTION",
]

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")

if not api_key:
    print("❌ 未找到 OPENAI_API_KEY，请在 .env 文件中设置")
    sys.exit(1)

print(f"API Key: {api_key[:10]}...")
print(f"Base URL: {base_url or '默认'}")
print(f"Model: {model_name}")

client = AsyncOpenAI(api_key=api_key, base_url=base_url)

# ── 翻译函数 ───────────────────────────────────────────
TRANSLATION_PROMPT = """你是一个专业的客服翻译专家。请将以下英文客服对话翻译成地道的中文。

要求：
1. 保持原文的语气和礼貌程度
2. {{变量名}} 格式的占位符保留不翻译，如 {{Order Number}}、{{Customer Support Phone Number}}
3. 翻译结果只返回中文，不要加任何解释
4. 使用自然的客服用语

英文原文：
{text}

中文翻译："""


async def translate_text(text: str, semaphore: asyncio.Semaphore) -> str:
    """调用 LLM 翻译单条文本"""
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": TRANSLATION_PROMPT.format(text=text)}],
                temperature=0.3,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ 翻译失败: {e}")
            return text  # 失败时返回原文


async def translate_batch(items: list[dict], category: str) -> list[dict]:
    """并发翻译一个分类的批次"""
    semaphore = asyncio.Semaphore(10)  # 最多 10 个并发
    tasks = []
    for item in items:
        tasks.append(translate_text(item["response"], semaphore))

    translated_responses = await asyncio.gather(*tasks)

    results = []
    for item, translated in zip(items, translated_responses):
        results.append({
            "category": item["category"],
            "intent": item["intent"],
            "instruction_en": item["instruction"],
            "instruction_cn": "",  # 后面单独翻译问题
            "response_en": item["response"],
            "response_cn": translated,
        })
    return results


async def translate_questions(instructions: list[str]) -> list[str]:
    """批量翻译用户问题"""
    semaphore = asyncio.Semaphore(10)
    tasks = [translate_text(q, semaphore) for q in instructions]
    return await asyncio.gather(*tasks)


async def main():
    from datasets import load_dataset

    print("\n📥 加载 Bitext 客服数据集...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    print(f"   总样本数: {len(ds)}")

    all_results = []

    for cat in CATEGORIES:
        # 过滤当前分类
        cat_data = ds.filter(lambda x: x["category"] == cat)
        total = len(cat_data)
        sample_count = min(SAMPLE_PER_CATEGORY, total)
        samples = cat_data.shuffle(seed=42).select(range(sample_count))

        print(f"\n🔄 翻译分类 [{cat}] — {sample_count}/{total} 条...")

        items = [
            {
                "category": s["category"],
                "intent": s["intent"],
                "instruction": s["instruction"],
                "response": s["response"],
            }
            for s in samples
        ]

        # 翻译 responses
        results = await translate_batch(items, cat)

        # 翻译 instructions（用户问题）
        instructions = [r["instruction_en"] for r in results]
        print(f"   翻译 {len(instructions)} 条用户问题...")
        translated_instructions = await translate_questions(instructions)
        for r, t_inst in zip(results, translated_instructions):
            r["instruction_cn"] = t_inst

        # 保存分类文件
        output_file = OUTPUT_DIR / f"{cat.lower()}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"   ✅ 保存: {output_file} ({len(results)} 条)")

        all_results.extend(results)

    # 保存合并文件
    all_file = OUTPUT_DIR / "all_categories.json"
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 完成！共翻译 {len(all_results)} 条，保存到 {OUTPUT_DIR}/")
    print(f"   合并文件: {all_file}")


if __name__ == "__main__":
    asyncio.run(main())
