#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect Bitext customer support dataset"""
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datasets import load_dataset

print("Loading bitext customer support dataset...")
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

print(f"\n=== Basic Info ===")
print(f"Split: {list(ds.keys())}")
print(f"Train size: {len(ds['train'])}")
print(f"Columns: {ds['train'].column_names}")
print(f"Features: {ds['train'].features}")

print(f"\n=== First 10 samples ===")
for i in range(min(10, len(ds['train']))):
    row = ds['train'][i]
    print(f"\n--- Sample {i+1} ---")
    for k, v in row.items():
        val_str = str(v)[:300] if v else "None"
        print(f"  {k}: {val_str}")

# Category distribution
print(f"\n=== Category Distribution ===")
from collections import Counter
categories = Counter(ds['train']['category'])
for cat, cnt in categories.most_common(20):
    print(f"  {cat}: {cnt}")

# Intent distribution (top 30)
print(f"\n=== Top 30 Intents ===")
intents = Counter(ds['train']['intent'])
for intent, cnt in intents.most_common(30):
    print(f"  {intent}: {cnt}")

# Instruction length stats
print(f"\n=== Instruction Length Stats ===")
lengths = [len(str(x)) for x in ds['train']['instruction']]
print(f"  Min: {min(lengths)} chars")
print(f"  Max: {max(lengths)} chars")
print(f"  Avg: {sum(lengths)/len(lengths):.0f} chars")

# Response length stats
print(f"\n=== Response Length Stats ===")
resp_lengths = [len(str(x)) for x in ds['train']['response']]
print(f"  Min: {min(resp_lengths)} chars")
print(f"  Max: {max(resp_lengths)} chars")
print(f"  Avg: {sum(resp_lengths)/len(resp_lengths):.0f} chars")
