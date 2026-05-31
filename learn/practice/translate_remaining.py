#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
续翻脚本：
1. 翻译缺失的分类（CONTACT, FEEDBACK, SUBSCRIPTION）
2. 重试之前翻译失败的条目（response_cn 为空的）
使用串行 + 延时避免 429
"""

import os
import json
import asyncio
import sys
import io
import time
from pathlib import Path
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv(Path(__file__).parent / ".env")

from openai import AsyncOpenAI

SAMPLE_PER_CATEGORY = 100
OUTPUT_DIR = Path(__file__).parent / "bitext_cn"

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")

client = AsyncOpenAI(api_key=api_key, base_url=base_url)

TRANSLATE_SYSTEM = """你是一个专业的客服翻译专家。将英文客服对话翻译成地道中文。

规则：
1. {{变量名}} 格式占位符保留不翻译
2. 保持礼貌专业的客服语气
3. 只返回中文翻译，不加任何解释"""

async def translate_one(text: str, max_retries: int = 3) -> str:
    """翻译单条文本，带重试和退避"""
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": TRANSLATE_SYSTEM},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                wait = 5 * (attempt + 1)
                print(f"  ⏳ 限流，等待 {wait}s...")
                await asyncio.sleep(wait)
            elif "400" in msg and "security" in msg:
                print(f"  ⚠️ 内容审核跳过: {text[:50]}...")
                return f"[内容审核跳过] {text}"
            else:
                print(f"  ⚠️ 错误: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
    return text  # 全部失败返回原文


async def translate_category(cat: str, samples: list[dict]):
    """翻译一个分类的所有样本"""
    output_file = OUTPUT_DIR / f"{cat.lower()}.json"

    # 如果已有数据，加载并检查缺失
    existing = []
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  已有 {len(existing)} 条，检查缺失...")

        # 找出 response_cn 为空或标记为失败的
        need_fix = [i for i, item in enumerate(existing)
                    if not item.get("response_cn") or
                    item["response_cn"].startswith("[") or
                    item["response_cn"].startswith("Sorry")]
        if need_fix:
            print(f"  需要修复 {len(need_fix)} 条...")
            for idx in need_fix:
                item = existing[idx]
                print(f"    修复 #{idx}: {item['intent']}")
                item["response_cn"] = await translate_one(item["response_en"])
                if not item.get("instruction_cn") or item["instruction_cn"].startswith("["):
                    item["instruction_cn"] = await translate_one(item["instruction_en"])
                await asyncio.sleep(0.5)  # 避免限流

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"  ✅ 修复完成: {output_file}")

        if len(existing) >= SAMPLE_PER_CATEGORY:
            return existing

    # 全新翻译
    if not samples:
        return existing

    print(f"  翻译 {cat}: {len(samples)} 条...")
    results = []

    for i, item in enumerate(samples):
        response_cn = await translate_one(item["response"])
        instruction_cn = await translate_one(item["instruction"])

        results.append({
            "category": item["category"],
            "intent": item["intent"],
            "instruction_en": item["instruction"],
            "instruction_cn": instruction_cn,
            "response_en": item["response"],
            "response_cn": response_cn,
        })

        # 每 10 条暂停一下
        if (i + 1) % 10 == 0:
            print(f"    进度: {i+1}/{len(samples)}")
            await asyncio.sleep(1)
        else:
            await asyncio.sleep(0.3)  # 每条间隔 0.3s

    # 合并已有和新翻译
    all_results = existing + results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 保存: {output_file} ({len(all_results)} 条)")
    return all_results


async def main():
    from datasets import load_dataset

    print("📥 加载数据集...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

    # 需要处理的分类
    all_categories = ["ORDER", "REFUND", "ACCOUNT", "PAYMENT", "DELIVERY",
                      "SHIPPING", "INVOICE", "CONTACT", "FEEDBACK", "SUBSCRIPTION"]

    grand_total = []

    for cat in all_categories:
        output_file = OUTPUT_DIR / f"{cat.lower()}.json"

        # 检查已有数据
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            # 统计有效翻译数
            valid = [x for x in existing
                     if x.get("response_cn") and
                     not x["response_cn"].startswith("[") and
                     not x["response_cn"].startswith("Sorry")]
            print(f"\n[{cat}] 已有 {len(existing)} 条，有效 {len(valid)} 条")
        else:
            existing = []
            valid = []
            print(f"\n[{cat}] 无已有数据")

        # 如果有效数据不足 100，补充翻译
        if len(valid) < SAMPLE_PER_CATEGORY:
            cat_data = ds.filter(lambda x: x["category"] == cat)
            needed = SAMPLE_PER_CATEGORY - len(valid)
            needed = min(needed, len(cat_data))
            samples = cat_data.shuffle(seed=42).select(range(needed))

            items = [
                {"category": s["category"], "intent": s["intent"],
                 "instruction": s["instruction"], "response": s["response"]}
                for s in samples
            ]
            await translate_category(cat, items)
        else:
            print(f"  ✅ 已有足够数据，跳过")

        # 读取最终数据
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                final = json.load(f)
            grand_total.extend(final)

    # 合并所有
    all_file = OUTPUT_DIR / "all_categories.json"
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(grand_total, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 全部完成！共 {len(grand_total)} 条，保存到 {all_file}")

    # 统计
    cats = {}
    for item in grand_total:
        c = item["category"]
        cats[c] = cats.get(c, 0) + 1
    print("\n各分类统计:")
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n} 条")


if __name__ == "__main__":
    asyncio.run(main())
