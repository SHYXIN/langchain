#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample diverse examples from Bitext dataset for translation planning"""
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datasets import load_dataset
from collections import defaultdict

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

# Pick 2 samples per category (10 categories x 2 = 20 samples)
samples_by_cat = defaultdict(list)
for row in ds:
    cat = row['category']
    if len(samples_by_cat[cat]) < 2:
        samples_by_cat[cat].append(row)

print("=== 2 samples per category (for translation reference) ===\n")
for cat in sorted(samples_by_cat.keys()):
    print(f"\n{'='*60}")
    print(f"Category: {cat}")
    print(f"{'='*60}")
    for row in samples_by_cat[cat]:
        print(f"\n  Intent: {row['intent']}")
        print(f"  Instruction: {row['instruction']}")
        resp = row['response']
        if len(resp) > 400:
            resp = resp[:400] + "..."
        print(f"  Response: {resp}")

# Also show intent diversity
print(f"\n\n=== All {len(set(ds['intent']))} unique intents ===")
all_intents = sorted(set(ds['intent']))
for i, intent in enumerate(all_intents):
    print(f"  {i+1:2d}. {intent}")
