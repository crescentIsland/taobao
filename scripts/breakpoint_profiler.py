"""
breakpoint_profiler_v2.py - 修正版
强制最小测量时长，消除系统时钟误差
"""

import time
import joblib
import numpy as np
import pandas as pd
import os

MODEL_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\model_comparison_40features'
FEATURES_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr1'

print("加载模型...")
lr = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_40f.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_40f.pkl'))

test_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'test_features.parquet'))
feature_cols = [c for c in test_df.columns if c not in ['user_id', 'item_id', 'item_category', 'label', 'predict_date']]
X = test_df[feature_cols]
X_scaled = scaler.transform(X)

# 预热
for _ in range(100):
    lr.predict_proba(X_scaled[0:1])
    lr.predict_proba(X_scaled[0:64])

print("\n" + "=" * 60)
print("修正版计时 - 强制最小测量时长")
print("=" * 60)


def measure_with_min_time(func, min_duration=1.0):
    """
    强制测量至少min_duration秒（默认1秒），减少系统误差
    """
    start = time.perf_counter()
    count = 0

    # 先跑1次
    func()
    count += 1

    # 继续跑直到超过min_duration
    while (time.perf_counter() - start) < min_duration:
        func()
        count += 1

    elapsed = time.perf_counter() - start
    return elapsed, count, elapsed / count * 1000  # 返回总时间、次数、每次毫秒


# 1. 单样本测试
print("\n[单样本推理 - 重复执行直到累计>1秒]")


def single_sample_task():
    for i in range(100):  # 每轮100次
        lr.predict_proba(X_scaled[i:i + 1])


elapsed, count, avg_ms = measure_with_min_time(single_sample_task, min_duration=2.0)
single_per_sample = avg_ms / 100  # 每次task是100个样本
print(f"测试{count}轮 x 100次 = {count * 100}次")
print(f"总耗时: {elapsed:.3f}s")
print(f"平均每样本: {single_per_sample:.4f} ms")

# 2. 批量测试 (batch=64)
print("\n[批量推理 Batch=64 - 重复执行直到累计>1秒]")


def batch_task():
    lr.predict_proba(X_scaled[0:64])


elapsed, count, batch_total_ms = measure_with_min_time(batch_task, min_duration=2.0)
batch_per_sample = batch_total_ms / 64
print(f"测试{count}轮 x 64样本 = {count * 64}样本")
print(f"总耗时: {elapsed:.3f}s")
print(f"每批次总耗时: {batch_total_ms:.4f} ms")
print(f"平均每样本: {batch_per_sample:.4f} ms")

# 3. 计算理论开销
python_overhead = single_per_sample - batch_per_sample  # 估算Python层开销
print(f"\n" + "=" * 60)
print("分析结果")
print("=" * 60)
print(f"单样本: {single_per_sample:.3f} ms")
print(f"  - Python层开销（函数调用等）: 估计 {python_overhead:.3f} ms")
print(f"  - 核心计算: 估计 {batch_per_sample:.3f} ms")
print(f"Batch64每样本: {batch_per_sample:.3f} ms")
print(f"实际加速比: {single_per_sample / batch_per_sample:.1f}x")
print(f"理论上限（仅核心计算）: {64 * batch_per_sample:.2f} ms可处理64个样本")