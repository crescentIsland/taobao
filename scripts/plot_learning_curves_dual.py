"""
双模型学习曲线绘制 - Logistic Regression vs XGBoost
修复版：解决XGBoost eval_metric重复定义问题
"""

import os
import sys
import time
import psutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
from tqdm import tqdm
import warnings
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 路径配置
project_root = r'E:\PycharmeProjects\taobao_user_behavior_analysis'
MODEL_DIR = os.path.join(project_root, 'results', 'model_comparison_40features')
FEATURES_DIR = os.path.join(project_root, 'data', 'features_for_lr1')
CHART_DIR = os.path.join(project_root, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)


def load_models_and_data():
    """加载模型和真实数据"""
    print("=" * 60)
    print("加载模型和数据...")

    # 加载模型
    lr_model = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_40f.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_40f.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_40f.pkl'))

    # 加载数据
    train_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'train_features.parquet'))
    val_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'val_features.parquet'))
    test_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'test_features.parquet'))

    # 准备特征
    feature_cols = [c for c in train_df.columns if
                    c not in ['user_id', 'item_id', 'item_category', 'label', 'predict_date']]

    X_train, y_train = train_df[feature_cols], train_df['label']
    X_val, y_val = val_df[feature_cols], val_df['label']
    X_test, y_test = test_df[feature_cols], test_df['label']

    # LR需要标准化
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"数据加载完成：训练集{len(train_df)}，验证集{len(val_df)}，测试集{len(test_df)}")
    print(f"特征数：{len(feature_cols)}")

    return (lr_model, xgb_model, scaler,
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_scaled, X_val_scaled, X_test_scaled,
            feature_cols)


def plot_sample_size_learning_curves(lr_model, xgb_model,
                                     X_train, y_train, X_val, y_val,
                                     X_train_scaled, X_val_scaled, feature_cols):
    """
    绘制随训练样本大小变化的学习曲线（双模型对比）
    """
    print("\n绘制样本大小学习曲线...")

    # 定义训练样本比例
    train_sizes = np.linspace(0.1, 1.0, 10)

    lr_train_scores, lr_val_scores = [], []
    xgb_train_scores, xgb_val_scores = [], []

    print("计算不同样本量下的性能...")
    for frac in tqdm(train_sizes, desc="测试样本比例"):
        n_samples = int(len(X_train) * frac)
        if n_samples < 100:
            continue

        # 取样
        X_subset = X_train.iloc[:n_samples]
        y_subset = y_train.iloc[:n_samples]
        X_subset_scaled = X_train_scaled[:n_samples]

        # 1. Logistic Regression
        lr_pred_train = lr_model.predict_proba(X_subset_scaled)[:, 1]
        lr_pred_val = lr_model.predict_proba(X_val_scaled)[:, 1]
        lr_train_scores.append(roc_auc_score(y_subset, lr_pred_train))
        lr_val_scores.append(roc_auc_score(y_val, lr_pred_val))

        # 2. XGBoost
        temp_xgb = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        temp_xgb.fit(X_subset, y_subset, verbose=False)

        xgb_pred_train = temp_xgb.predict_proba(X_subset)[:, 1]
        xgb_pred_val = temp_xgb.predict_proba(X_val)[:, 1]
        xgb_train_scores.append(roc_auc_score(y_subset, xgb_pred_train))
        xgb_val_scores.append(roc_auc_score(y_val, xgb_pred_val))

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Learning Curves Comparison: LR vs XGBoost (40 Features)', fontsize=16, fontweight='bold')

    # 图1: 样本大小学习曲线
    ax1 = axes[0, 0]
    ax1.plot(train_sizes * 100, lr_train_scores, 'o-', label='LR Train', color='blue', linewidth=2)
    ax1.plot(train_sizes * 100, lr_val_scores, 's-', label='LR Val', color='lightblue', linewidth=2)
    ax1.plot(train_sizes * 100, xgb_train_scores, 'o-', label='XGB Train', color='red', linewidth=2)
    ax1.plot(train_sizes * 100, xgb_val_scores, 's-', label='XGB Val', color='lightcoral', linewidth=2)
    ax1.set_xlabel('Training Set Size (%)')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Learning Curves (Sample Size vs AUC)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 过拟合分析
    ax2 = axes[0, 1]
    lr_gaps = [t - v for t, v in zip(lr_train_scores, lr_val_scores)]
    xgb_gaps = [t - v for t, v in zip(xgb_train_scores, xgb_val_scores)]

    ax2.plot(train_sizes * 100, lr_gaps, 'o-', label='LR Gap', color='blue', linewidth=2)
    ax2.plot(train_sizes * 100, xgb_gaps, 's-', label='XGB Gap', color='red', linewidth=2)
    ax2.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Warning (2%)')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting (5%)')
    ax2.fill_between(train_sizes * 100, 0, 0.02, alpha=0.2, color='green', label='Good Zone')
    ax2.set_xlabel('Training Set Size (%)')
    ax2.set_ylabel('AUC Gap (Train - Val)')
    ax2.set_title('Overfitting Analysis (Generalization Gap)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 最终性能对比
    ax3 = axes[1, 0]
    final_metrics = {
        'LR Train': lr_train_scores[-1],
        'LR Val': lr_val_scores[-1],
        'XGB Train': xgb_train_scores[-1],
        'XGB Val': xgb_val_scores[-1]
    }
    colors = ['skyblue', 'lightblue', 'lightcoral', 'salmon']
    bars = ax3.bar(final_metrics.keys(), final_metrics.values(), color=colors, alpha=0.8)
    ax3.set_ylabel('AUC Score')
    ax3.set_title('Final Performance (100% Training Data)')
    ax3.set_ylim([0.7, 0.9])
    ax3.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    # 图4: 数据效率
    ax4 = axes[1, 1]
    target_lr = 0.9 * lr_val_scores[-1]
    target_xgb = 0.9 * xgb_val_scores[-1]

    lr_90_idx = next((i for i, score in enumerate(lr_val_scores) if score >= target_lr), len(lr_val_scores) - 1)
    xgb_90_idx = next((i for i, score in enumerate(xgb_val_scores) if score >= target_xgb), len(xgb_val_scores) - 1)

    efficiency_data = {
        'LR (90%性能)': train_sizes[lr_90_idx] * 100,
        'XGB (90%性能)': train_sizes[xgb_90_idx] * 100,
    }

    bars = ax4.bar(efficiency_data.keys(), efficiency_data.values(),
                   color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Training Data Needed (%)')
    ax4.set_title('Data Efficiency (90% Final Performance)')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, 'learning_curves_LR_vs_XGB.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ 学习曲线对比图已保存: {chart_path}")
    plt.close()

    # 保存数据
    curve_data = pd.DataFrame({
        'train_size_pct': train_sizes * 100,
        'lr_train_auc': lr_train_scores,
        'lr_val_auc': lr_val_scores,
        'lr_gap': lr_gaps,
        'xgb_train_auc': xgb_train_scores,
        'xgb_val_auc': xgb_val_scores,
        'xgb_gap': xgb_gaps
    })
    csv_path = os.path.join(CHART_DIR, 'learning_curve_data.csv')
    curve_data.to_csv(csv_path, index=False)
    print(f"✓ 学习曲线数据已保存: {csv_path}")

    return curve_data


def plot_xgb_iterations(X_train, y_train, X_val, y_val):  # <-- 修复：移除xgb_model参数，重新训练
    """
    绘制XGBoost的迭代学习曲线
    修复：避免eval_metric重复定义，只保留构造函数中的设置
    """
    print("\n绘制XGBoost迭代曲线...")

    # 重新训练模型，只在构造函数中设置eval_metric
    xgb_iter = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'  # <-- 只在这里设置，不在fit中重复设置
    )

    # fit中不再传eval_metric
    xgb_iter.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False  # <-- 修复：移除eval_metric参数
    )

    # 获取评估结果
    results = xgb_iter.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('XGBoost Training Dynamics (40 Features)', fontsize=14, fontweight='bold')

    # 图1: 迭代AUC变化
    ax1 = axes[0]
    ax1.plot(x_axis, results['validation_0']['auc'], label='Train AUC', color='blue', linewidth=2)
    ax1.plot(x_axis, results['validation_1']['auc'], label='Val AUC', color='red', linewidth=2)
    ax1.set_xlabel('Iterations (Trees)')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('AUC vs Training Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 标记最佳迭代点
    best_iter = np.argmax(results['validation_1']['auc'])
    best_auc = results['validation_1']['auc'][best_iter]
    ax1.axvline(x=best_iter, color='green', linestyle='--', alpha=0.7)
    ax1.scatter([best_iter], [best_auc], color='green', s=100, zorder=5)
    ax1.text(best_iter, best_auc + 0.01, f'Best@{best_iter}\nAUC={best_auc:.4f}',
             ha='center', fontsize=9, color='green')

    # 图2: 过拟合累积
    ax2 = axes[1]
    gaps = [t - v for t, v in zip(results['validation_0']['auc'], results['validation_1']['auc'])]
    ax2.plot(x_axis, gaps, color='purple', linewidth=2)
    ax2.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Warning (2%)')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting (5%)')
    ax2.fill_between(x_axis, 0, 0.02, alpha=0.2, color='green', label='Good Zone')
    ax2.set_xlabel('Iterations (Trees)')
    ax2.set_ylabel('AUC Gap (Train - Val)')
    ax2.set_title('Overfitting Accumulation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, 'xgboost_iterations.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ XGBoost迭代曲线已保存: {chart_path}")
    print(f"  最佳迭代轮数: {best_iter}, 最佳验证AUC: {best_auc:.4f}")
    plt.close()


def plot_feature_importance_comparison(lr_model, xgb_model, feature_cols):
    """
    绘制两个模型的特征重要性对比
    """
    print("\n绘制特征重要性对比...")

    # LR系数（绝对值）
    lr_importance = np.abs(lr_model.coef_[0])
    # XGB重要性
    xgb_importance = xgb_model.feature_importances_

    # 创建对比DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'lr_importance': lr_importance / np.max(lr_importance),
        'xgb_importance': xgb_importance / np.max(xgb_importance)
    })

    # 按XGB重要性排序
    importance_df = importance_df.sort_values('xgb_importance', ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(importance_df))
    width = 0.35

    ax.barh(y_pos - width / 2, importance_df['lr_importance'], width,
            label='Logistic Regression (|coef|)', color='skyblue', alpha=0.8)
    ax.barh(y_pos + width / 2, importance_df['xgb_importance'], width,
            label='XGBoost', color='lightcoral', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'], fontsize=9)
    ax.set_xlabel('Normalized Importance')
    ax.set_title('Top 20 Feature Importance: LR vs XGBoost (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, 'feature_importance_comparison.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ 特征重要性对比图已保存: {chart_path}")
    plt.close()

    # 计算特征重要性相关性
    corr, p_value = spearmanr(importance_df['lr_importance'], importance_df['xgb_importance'])
    print(f"  特征重要性Spearman相关性: {corr:.3f} (p={p_value:.3f})")
    if corr > 0.7:
        print("  ✓ 两个模型高度关注相同特征")
    elif corr > 0.4:
        print("  △ 两个模型关注部分相同特征")
    else:
        print("  ✗ 两个模型关注不同特征")


def main():
    print("=" * 70)
    print("双模型学习曲线绘制 - LR vs XGBoost")
    print("=" * 70)

    # 加载数据
    (lr_model, xgb_model, scaler,
     X_train, y_train, X_val, y_val, X_test, y_test,
     X_train_scaled, X_val_scaled, X_test_scaled,
     feature_cols) = load_models_and_data()

    # 1. 样本大小学习曲线
    curve_data = plot_sample_size_learning_curves(
        lr_model, xgb_model,
        X_train, y_train, X_val, y_val,
        X_train_scaled, X_val_scaled, feature_cols
    )

    # 2. XGBoost迭代曲线 - 修复：不传xgb_model，直接传数据重新训练
    plot_xgb_iterations(X_train, y_train, X_val, y_val)  # <-- 修复：简化参数

    # 3. 特征重要性对比
    plot_feature_importance_comparison(lr_model, xgb_model, feature_cols)

    print("\n" + "=" * 70)
    print("所有图表生成完成！")
    print(f"图表保存位置: {CHART_DIR}")
    print("生成文件:")
    print("  1. learning_curves_LR_vs_XGB.png - 样本大小学习曲线对比")
    print("  2. xgboost_iterations.png - XGBoost迭代曲线")
    print("  3. feature_importance_comparison.png - 特征重要性对比")
    print("  4. learning_curve_data.csv - 原始数据")
    print("=" * 70)


if __name__ == "__main__":
    main()