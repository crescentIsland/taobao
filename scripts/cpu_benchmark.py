"""
纯CPU环境模型性能测试
测试Logistic Regression和XGBoost在CPU上的推理速度
"""

import os
import time
import psutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# 路径配置（修改为你的实际路径）
MODEL_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\model_comparison_40features'


def load_models():
    lr = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_40f.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgboost_40f.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_40f.pkl'))
    return lr, xgb, scaler


def test_speed(model, X_data, model_name, scaler=None, is_lr=True):
    """测试推理速度"""
    print(f"\n{'=' * 60}")
    print(f"测试 {model_name} (CPU模式)")
    print(f"{'=' * 60}")

    # 数据准备
    X_processed = scaler.transform(X_data) if (is_lr and scaler) else X_data

    # 预热
    _ = model.predict_proba(X_processed[:5])

    results = {}

    # 1. 单样本测试 (模拟API)
    n = 200
    start = time.perf_counter()
    for i in range(n):
        sample = X_processed[i:i + 1] if is_lr else X_data.iloc[i:i + 1]
        _ = model.predict_proba(sample)
    single_time = (time.perf_counter() - start) / n * 1000  # ms
    results['single_ms'] = single_time
    print(f"单样本推理: {single_time:.2f} ms/样本")

    # 2. 批量测试 batch=32
    batch_size = 32
    n_batches = 50
    start = time.perf_counter()
    for i in range(n_batches):
        batch = X_processed[i * batch_size:(i + 1) * batch_size] if is_lr else X_data.iloc[
                                                                               i * batch_size:(i + 1) * batch_size]
        _ = model.predict_proba(batch)
    batch32_time = (time.perf_counter() - start) / (n_batches * batch_size) * 1000
    results['batch32_ms'] = batch32_time
    print(f"Batch=32: {batch32_time:.2f} ms/样本 | 单次批次: {(time.perf_counter() - start) / n_batches * 1000:.1f}ms")

    # 3. 批量测试 batch=64
    batch_size = 64
    n_batches = 25
    start = time.perf_counter()
    for i in range(n_batches):
        batch = X_processed[i * batch_size:(i + 1) * batch_size] if is_lr else X_data.iloc[
                                                                               i * batch_size:(i + 1) * batch_size]
        _ = model.predict_proba(batch)
    batch64_time = (time.perf_counter() - start) / (n_batches * batch_size) * 1000
    results['batch64_ms'] = batch64_time
    print(f"Batch=64: {batch64_time:.2f} ms/样本 | 单次批次: {(time.perf_counter() - start) / n_batches * 1000:.1f}ms")

    # 4. 资源监控
    process = psutil.Process()
    cpu_percent = process.cpu_percent()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"CPU占用: {cpu_percent:.1f}% | 内存: {mem_mb:.1f}MB")
    results['cpu'] = cpu_percent
    results['memory'] = mem_mb

    return results


def get_file_size():
    """获取模型文件大小"""
    files = {
        'Logistic Regression': 'logistic_regression_40f.pkl',
        'XGBoost': 'xgboost_40f.pkl',
        'Scaler': 'scaler_40f.pkl'
    }
    for name, f in files.items():
        path = os.path.join(MODEL_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"{name}: {size:.2f} MB")


def main():
    print("模型性能测试 (纯CPU环境)")
    print(f"测试时间: {datetime.now()}")
    print(f"CPU: {psutil.cpu_count()}核 {psutil.cpu_freq().current:.0f}MHz")

    # 显示模型大小
    print("\n模型文件大小:")
    get_file_size()

    # 加载模型
    lr_model, xgb_model, scaler = load_models()

    # 生成测试数据 (40维特征)
    np.random.seed(42)
    X_test = pd.DataFrame(np.random.randn(5000, 40), columns=[f'f{i}' for i in range(40)])

    # 测试
    lr_res = test_speed(lr_model, X_test, "Logistic Regression", scaler, True)
    xgb_res = test_speed(xgb_model, X_test, "XGBoost", None, False)

    # 生成报告
    print(f"\n{'=' * 60}")
    print("测试完成 - 请复制以下结果到文档:")
    print(f"{'=' * 60}")
    print(f"""
Logistic Regression (CPU):
- 单样本: {lr_res['single_ms']:.2f} ms
- Batch=32: {lr_res['batch32_ms']:.2f} ms/样本
- Batch=64: {lr_res['batch64_ms']:.2f} ms/样本
- CPU占用: {lr_res['cpu']:.1f}%
- 内存: {lr_res['memory']:.1f} MB

XGBoost (CPU):
- 单样本: {xgb_res['single_ms']:.2f} ms  
- Batch=32: {xgb_res['batch32_ms']:.2f} ms/样本
- Batch=64: {xgb_res['batch64_ms']:.2f} ms/样本
- CPU占用: {xgb_res['cpu']:.1f}%
- 内存: {xgb_res['memory']:.1f} MB

速度对比:
- 单样本: XGBoost比LR慢 {xgb_res['single_ms'] / lr_res['single_ms']:.1f} 倍
- Batch=64: XGBoost比LR慢 {xgb_res['batch64_ms'] / lr_res['batch64_ms']:.1f} 倍
    """)


if __name__ == "__main__":
    main()