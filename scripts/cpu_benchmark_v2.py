"""
纯CPU环境模型性能测试 - 修复版（使用真实特征数据）
"""

import os
import time
import psutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ==================== 配置路径 ====================
MODEL_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\model_comparison_40features'
FEATURES_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr1'


def load_models():
    """加载模型"""
    print("加载模型...")
    lr = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_40f.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgboost_40f.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_40f.pkl'))
    print("模型加载完成")
    return lr, xgb, scaler


def load_real_data():
    """加载真实的特征数据"""
    print(f"从 {FEATURES_DIR} 加载测试数据...")
    try:
        # 尝试加载parquet
        test_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'test_features.parquet'))
        print(f"加载成功: {len(test_df)} 样本, {len(test_df.columns)} 列")
        return test_df
    except:
        try:
            test_df = pd.read_csv(os.path.join(FEATURES_DIR, 'test_features.csv'))
            print(f"加载成功: {len(test_df)} 样本")
            return test_df
        except Exception as e:
            print(f"加载失败: {e}")
            return None


def get_model_size():
    """获取模型文件大小"""
    files = {
        'Logistic Regression': 'logistic_regression_40f.pkl',
        'XGBoost': 'xgboost_40f.pkl',
        'Scaler': 'scaler_40f.pkl'
    }
    sizes = {}
    print("\n模型文件大小:")
    for name, f in files.items():
        path = os.path.join(MODEL_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            sizes[name] = size
            print(f"  {name}: {size:.2f} MB")
        else:
            print(f"  [!] 未找到: {path}")
            sizes[name] = 0
    return sizes


def benchmark_speed(model, X_data, model_name, scaler=None, is_lr=True):
    """测试推理速度"""
    print(f"\n{'=' * 60}")
    print(f"测试 {model_name} (CPU模式)")
    print(f"{'=' * 60}")

    results = {}

    # 数据准备（LR需要标准化）
    if is_lr and scaler is not None:
        X_processed = scaler.transform(X_data)
        print(f"数据已标准化，形状: {X_processed.shape}")
    else:
        X_processed = X_data
        print(f"原始数据，形状: {X_data.shape}")

    # 确保数据足够
    n_samples = min(5000, len(X_processed))
    if is_lr:
        X_test = X_processed[:n_samples]
    else:
        X_test = X_data.iloc[:n_samples] if hasattr(X_data, 'iloc') else X_data[:n_samples]

    # 1. 单样本测试 (模拟API)
    print("测试单样本推理...")
    n = min(200, n_samples)
    start = time.perf_counter()
    for i in range(n):
        if is_lr:
            _ = model.predict_proba(X_test[i:i + 1])
        else:
            sample = X_test.iloc[i:i + 1] if hasattr(X_test, 'iloc') else X_test[i:i + 1]
            _ = model.predict_proba(sample)
    single_time = (time.perf_counter() - start) / n * 1000  # ms
    results['single_ms'] = single_time
    print(f"  单样本: {single_time:.2f} ms/样本 (测试{n}次)")

    # 2. 批量测试 batch=32
    print("测试 Batch=32...")
    batch_size = 32
    n_batches = min(50, n_samples // batch_size)
    start = time.perf_counter()
    for i in range(n_batches):
        if is_lr:
            batch = X_test[i * batch_size:(i + 1) * batch_size]
        else:
            batch = X_test.iloc[i * batch_size:(i + 1) * batch_size] if hasattr(X_test, 'iloc') else X_test[
                                                                                                     i * batch_size:(
                                                                                                                                i + 1) * batch_size]
        _ = model.predict_proba(batch)
    batch32_total = time.perf_counter() - start
    batch32_per_sample = batch32_total / (n_batches * batch_size) * 1000
    results['batch32_ms'] = batch32_per_sample
    print(f"  Batch=32: {batch32_per_sample:.2f} ms/样本 | 批次耗时: {batch32_total / n_batches * 1000:.1f}ms")

    # 3. 批量测试 batch=64
    print("测试 Batch=64...")
    batch_size = 64
    n_batches = min(25, n_samples // batch_size)
    start = time.perf_counter()
    for i in range(n_batches):
        if is_lr:
            batch = X_test[i * batch_size:(i + 1) * batch_size]
        else:
            batch = X_test.iloc[i * batch_size:(i + 1) * batch_size] if hasattr(X_test, 'iloc') else X_test[
                                                                                                     i * batch_size:(
                                                                                                                                i + 1) * batch_size]
        _ = model.predict_proba(batch)
    batch64_total = time.perf_counter() - start
    batch64_per_sample = batch64_total / (n_batches * batch_size) * 1000
    results['batch64_ms'] = batch64_per_sample
    print(f"  Batch=64: {batch64_per_sample:.2f} ms/样本 | 批次耗时: {batch64_total / n_batches * 1000:.1f}ms")

    # 4. 资源监控
    process = psutil.Process()
    cpu_percent = process.cpu_percent()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"\n资源占用: CPU {cpu_percent:.1f}% | 内存 {mem_mb:.1f}MB")
    results['cpu'] = cpu_percent
    results['memory'] = mem_mb

    return results


def main():
    print("=" * 60)
    print("CPU-only模型性能测试 (使用真实数据)")
    print(f"测试时间: {datetime.now()}")
    print(f"CPU: {psutil.cpu_count()}核 {psutil.cpu_freq().current:.0f}MHz")
    print(f"内存: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
    print("=" * 60)

    # 获取模型大小
    sizes = get_model_size()

    # 加载模型
    try:
        lr_model, xgb_model, scaler = load_models()
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 加载真实数据
    test_df = load_real_data()
    if test_df is None:
        print("无法加载测试数据，退出")
        return

    # 分离特征和标签
    feature_cols = [c for c in test_df.columns if
                    c not in ['user_id', 'item_id', 'item_category', 'label', 'predict_date']]
    print(f"特征列: {len(feature_cols)}个")
    print(f"前5个特征: {feature_cols[:5]}")

    X_test = test_df[feature_cols]
    y_test = test_df['label'] if 'label' in test_df.columns else None

    # 测试LR
    lr_results = benchmark_speed(lr_model, X_test, "Logistic Regression", scaler, True)

    # 测试XGB
    xgb_results = benchmark_speed(xgb_model, X_test, "XGBoost", None, False)

    # 输出最终结果
    print("\n" + "=" * 60)
    print("测试结果汇总（直接复制到文档）")
    print("=" * 60)

    # 生成文档格式
    doc_content = f"""
【5.1 推理速度】

Logistic Regression (纯CPU):
- 单样本推理: {lr_results['single_ms']:.2f} ms
- 批量推理 (batch=32): {lr_results['batch32_ms']:.2f} ms/样本 (每批次约 {lr_results['batch32_ms'] * 32 / 1000:.2f}ms)
- 批量推理 (batch=64): {lr_results['batch64_ms']:.2f} ms/样本 (每批次约 {lr_results['batch64_ms'] * 64 / 1000:.2f}ms)
- 硬件环境: CPU {psutil.cpu_count()}核, {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB RAM, 无GPU

XGBoost (纯CPU):
- 单样本推理: {xgb_results['single_ms']:.2f} ms
- 批量推理 (batch=32): {xgb_results['batch32_ms']:.2f} ms/样本 (每批次约 {xgb_results['batch32_ms'] * 32 / 1000:.2f}ms)
- 批量推理 (batch=64): {xgb_results['batch64_ms']:.2f} ms/样本 (每批次约 {xgb_results['batch64_ms'] * 64 / 1000:.2f}ms)
- 硬件环境: 同上

速度对比: XGBoost单样本比LR慢 {xgb_results['single_ms'] / lr_results['single_ms']:.1f} 倍

【5.2 模型大小】
- Logistic Regression: {sizes.get('Logistic Regression', 0):.2f} MB (joblib .pkl格式)
- XGBoost: {sizes.get('XGBoost', 0):.2f} MB (joblib .pkl格式)
- 特征标准化器: {sizes.get('Scaler', 0):.2f} MB

【5.3 资源消耗 (纯CPU)】

Logistic Regression:
- CPU峰值: {lr_results['cpu']:.1f}%
- 内存占用: {lr_results['memory']:.1f} MB
- GPU显存: 不适用 (纯CPU计算)

XGBoost:
- CPU峰值: {xgb_results['cpu']:.1f}%
- 内存占用: {xgb_results['memory']:.1f} MB
- GPU显存: 不适用 (纯CPU计算)
"""
    print(doc_content)

    # 保存到文件
    output_file = os.path.join(MODEL_DIR, 'benchmark_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    print(f"\n✓ 结果已保存: {output_file}")


if __name__ == "__main__":
    main()