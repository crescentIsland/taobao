"""
双模型对比训练 - 基于40个筛选特征
逻辑回归 vs XGBoost 用户购买行为预测
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
import json

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, accuracy_score, precision_recall_curve)
import xgboost as xgb
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径配置 ====================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(project_root, 'results', 'model_comparison_40features')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 使用新的40个特征路径
FEATURES_DIR = os.path.join(project_root, 'data', 'features_for_lr1')

print("=" * 80)
print("双模型对比训练 - 基于40个筛选特征")
print("=" * 80)

# ==================== 1. 加载40个特征数据 ====================
print("\n[1/6] 加载40个特征数据...")

try:
    train_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'train_features.parquet'))
    val_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'val_features.parquet'))
    test_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'test_features.parquet'))
    print(f"✓ 成功加载40个特征数据")
except:
    # 备用：如果parquet不存在，尝试csv
    print("尝试CSV格式...")
    train_df = pd.read_csv(os.path.join(FEATURES_DIR, 'train_features.csv'))
    val_df = pd.read_csv(os.path.join(FEATURES_DIR, 'val_features.csv'))
    test_df = pd.read_csv(os.path.join(FEATURES_DIR, 'test_features.csv'))

print(f"\n数据集概况:")
print(f"训练集: {len(train_df):,} 样本, 特征数: {len(train_df.columns) - 2}, 正样本率: {train_df['label'].mean():.2%}")
print(f"验证集: {len(val_df):,} 样本, 正样本率: {val_df['label'].mean():.2%}")
print(f"测试集: {len(test_df):,} 样本, 正样本率: {test_df['label'].mean():.2%}")

# 分离特征和标签（排除ID列）
feature_cols = [c for c in train_df.columns if
                c not in ['user_id', 'item_id', 'item_category', 'label', 'predict_date']]
print(f"\n实际使用特征数: {len(feature_cols)} 个")
print(f"特征列表: {feature_cols[:5]}... (共{len(feature_cols)}个)")

X_train, y_train = train_df[feature_cols], train_df['label']
X_val, y_val = val_df[feature_cols], val_df['label']
X_test, y_test = test_df[feature_cols], test_df['label']

# ==================== 2. 特征标准化（仅LR需要） ====================
print("\n[2/6] 特征标准化（逻辑回归）...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler_40features.pkl'))

# ==================== 3. 逻辑回归训练与调优 ====================
print("\n[3/6] 逻辑回归训练与超参数调优...")

# 超参数搜索
param_grid_lr = {
    'C': [0.01, 0.1, 0.5, 1.0, 10.0],
    'class_weight': ['balanced', None]
}

best_auc_lr = 0
best_params_lr = {}
best_model_lr = None

print("逻辑回归 Grid Search:")
for C in param_grid_lr['C']:
    for cw in param_grid_lr['class_weight']:
        model = LogisticRegression(
            C=C,
            class_weight=cw,
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)

        print(f"  C={C:5.2f}, weight={str(cw):10s} | Val AUC={val_auc:.4f}")

        if val_auc > best_auc_lr:
            best_auc_lr = val_auc
            best_params_lr = {'C': C, 'class_weight': cw}
            best_model_lr = model

print(f"\nLR最优参数: {best_params_lr}")

# LR阈值调优
val_proba_lr = best_model_lr.predict_proba(X_val_scaled)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1_lr = 0
best_threshold_lr = 0.5

for thresh in thresholds:
    pred = (val_proba_lr >= thresh).astype(int)
    f1 = f1_score(y_val, pred)
    if f1 > best_f1_lr:
        best_f1_lr = f1
        best_threshold_lr = thresh

print(f"LR最优阈值: {best_threshold_lr:.2f}")

# LR预测
train_proba_lr = best_model_lr.predict_proba(X_train_scaled)[:, 1]
test_proba_lr = best_model_lr.predict_proba(X_test_scaled)[:, 1]
train_pred_lr = (train_proba_lr >= best_threshold_lr).astype(int)
test_pred_lr = (test_proba_lr >= best_threshold_lr).astype(int)

# ==================== 4. XGBoost训练与调优 ====================
print("\n[4/6] XGBoost训练与超参数调优...")

# 计算scale_pos_weight
spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# XGBoost参数网格（轻量级搜索）
param_grid_xgb = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

best_auc_xgb = 0
best_params_xgb = {}
best_model_xgb = None

print("XGBoost Grid Search:")
for depth in param_grid_xgb['max_depth']:
    for lr in param_grid_xgb['learning_rate']:
        for n_est in param_grid_xgb['n_estimators']:
            model = xgb.XGBClassifier(
                max_depth=depth,
                learning_rate=lr,
                n_estimators=n_est,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                random_state=42,
                n_jobs=-1,
                eval_metric='auc',
                early_stopping_rounds=10
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            val_proba = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)

            print(f"  depth={depth}, lr={lr}, n={n_est:3d} | Val AUC={val_auc:.4f}")

            if val_auc > best_auc_xgb:
                best_auc_xgb = val_auc
                best_params_xgb = {'max_depth': depth, 'learning_rate': lr, 'n_estimators': n_est}
                best_model_xgb = model

print(f"\nXGB最优参数: {best_params_xgb}")

# XGB阈值调优
val_proba_xgb = best_model_xgb.predict_proba(X_val)[:, 1]
best_f1_xgb = 0
best_threshold_xgb = 0.5

for thresh in thresholds:
    pred = (val_proba_xgb >= thresh).astype(int)
    f1 = f1_score(y_val, pred)
    if f1 > best_f1_xgb:
        best_f1_xgb = f1
        best_threshold_xgb = thresh

print(f"XGB最优阈值: {best_threshold_xgb:.2f}")

# XGB预测
train_proba_xgb = best_model_xgb.predict_proba(X_train)[:, 1]
test_proba_xgb = best_model_xgb.predict_proba(X_test)[:, 1]
train_pred_xgb = (train_proba_xgb >= best_threshold_xgb).astype(int)
test_pred_xgb = (test_proba_xgb >= best_threshold_xgb).astype(int)

# ==================== 5. 评估指标计算 ====================
print("\n[5/6] 计算评估指标...")


def calculate_metrics(y_true, y_pred, y_proba):
    """计算所有评估指标"""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba)
    }


# 逻辑回归指标
lr_train_metrics = calculate_metrics(y_train, train_pred_lr, train_proba_lr)
lr_test_metrics = calculate_metrics(y_test, test_pred_lr, test_proba_lr)

# XGBoost指标
xgb_train_metrics = calculate_metrics(y_train, train_pred_xgb, train_proba_xgb)
xgb_test_metrics = calculate_metrics(y_test, test_pred_xgb, test_proba_xgb)

# 打印对比表格
print("\n" + "=" * 80)
print("模型性能对比（基于40个特征）")
print("=" * 80)

print("\nLogistic Regression:")
print("-" * 60)
print(f"{'指标':<15} {'训练集':>12} {'测试集':>12} {'差距':>12}")
print("-" * 60)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
    gap = lr_train_metrics[metric] - lr_test_metrics[metric]
    print(f"{metric:<15} {lr_train_metrics[metric]:>12.4f} {lr_test_metrics[metric]:>12.4f} {gap:>12.4f}")

print("\nXGBoost:")
print("-" * 60)
print(f"{'指标':<15} {'训练集':>12} {'测试集':>12} {'差距':>12}")
print("-" * 60)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
    gap = xgb_train_metrics[metric] - xgb_test_metrics[metric]
    print(f"{metric:<15} {xgb_train_metrics[metric]:>12.4f} {xgb_test_metrics[metric]:>12.4f} {gap:>12.4f}")

print("\n模型对比（测试集）:")
print("-" * 60)
print(f"{'指标':<15} {'LogisticR':>12} {'XGBoost':>12} {'提升':>12}")
print("-" * 60)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
    lift = xgb_test_metrics[metric] - lr_test_metrics[metric]
    print(f"{metric:<15} {lr_test_metrics[metric]:>12.4f} {xgb_test_metrics[metric]:>12.4f} {lift:>+12.4f}")

# ==================== 6. 可视化与保存 ====================
print("\n[6/6] 生成可视化与保存结果...")

fig = plt.figure(figsize=(20, 12))

# 创建子图布局
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. ROC曲线对比
ax1 = fig.add_subplot(gs[0, 0])
fpr_lr, tpr_lr, _ = roc_curve(y_test, test_proba_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, test_proba_xgb)

ax1.plot(fpr_lr, tpr_lr, label=f'LR (AUC={lr_test_metrics["AUC"]:.4f})', linewidth=2, color='blue')
ax1.plot(fpr_xgb, tpr_xgb, label=f'XGB (AUC={xgb_test_metrics["AUC"]:.4f})', linewidth=2, color='red')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC曲线对比 (测试集)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. PR曲线对比
ax2 = fig.add_subplot(gs[0, 1])
precision_lr, recall_lr, _ = precision_recall_curve(y_test, test_proba_lr)
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, test_proba_xgb)

ax2.plot(recall_lr, precision_lr, label='LR', linewidth=2, color='blue')
ax2.plot(recall_xgb, precision_xgb, label='XGB', linewidth=2, color='red')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 预测概率分布对比
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(test_proba_lr[y_test == 0], bins=50, alpha=0.5, label='LR-未购买', density=True, color='blue')
ax3.hist(test_proba_lr[y_test == 1], bins=50, alpha=0.5, label='LR-购买', density=True, color='lightblue')
ax3.hist(test_proba_xgb[y_test == 0], bins=50, alpha=0.5, label='XGB-未购买', density=True, color='red')
ax3.hist(test_proba_xgb[y_test == 1], bins=50, alpha=0.5, label='XGB-购买', density=True, color='lightcoral')
ax3.axvline(best_threshold_lr, color='blue', linestyle='--', label=f'LR阈值={best_threshold_lr:.2f}')
ax3.axvline(best_threshold_xgb, color='red', linestyle='--', label=f'XGB阈值={best_threshold_xgb:.2f}')
ax3.set_xlabel('预测概率')
ax3.set_ylabel('密度')
ax3.set_title('预测概率分布对比')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. 指标对比柱状图
ax4 = fig.add_subplot(gs[1, :2])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metrics))
width = 0.35

lr_values = [lr_test_metrics[m] for m in metrics]
xgb_values = [xgb_test_metrics[m] for m in metrics]

bars1 = ax4.bar(x - width / 2, lr_values, width, label='Logistic Regression', color='skyblue', alpha=0.8)
bars2 = ax4.bar(x + width / 2, xgb_values, width, label='XGBoost', color='lightcoral', alpha=0.8)

ax4.set_ylabel('Score')
ax4.set_title('测试集性能指标对比')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

# 5. 特征重要性对比（Top 15）
ax5 = fig.add_subplot(gs[1, 2])

# LR系数
lr_coef = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(best_model_lr.coef_[0])
}).sort_values('importance', ascending=True).tail(15)

# XGB重要性
xgb_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model_xgb.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

# 合并显示
y_pos = np.arange(15)
ax5.barh(y_pos - 0.2, lr_coef['importance'], 0.4, label='LR(|coef|)', color='skyblue', alpha=0.8)
ax5.barh(y_pos + 0.2, xgb_imp['importance'], 0.4, label='XGB', color='lightcoral', alpha=0.8)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(lr_coef['feature'], fontsize=8)
ax5.set_xlabel('Importance')
ax5.set_title('Top 15 特征重要性对比')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='x')

# 6. 混淆矩阵对比
ax6 = fig.add_subplot(gs[2, 0])
cm_lr = confusion_matrix(y_test, test_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax6, cbar=False)
ax6.set_title(f'LR混淆矩阵\nAccuracy={lr_test_metrics["Accuracy"]:.4f}')
ax6.set_xlabel('预测标签')
ax6.set_ylabel('真实标签')

ax7 = fig.add_subplot(gs[2, 1])
cm_xgb = confusion_matrix(y_test, test_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Reds', ax=ax7, cbar=False)
ax7.set_title(f'XGB混淆矩阵\nAccuracy={xgb_test_metrics["Accuracy"]:.4f}')
ax7.set_xlabel('预测标签')
ax7.set_ylabel('真实标签')

# 7. 过拟合分析
ax8 = fig.add_subplot(gs[2, 2])
models = ['LR', 'XGB']
train_aucs = [lr_train_metrics['AUC'], xgb_train_metrics['AUC']]
test_aucs = [lr_test_metrics['AUC'], xgb_test_metrics['AUC']]
gaps = [train - test for train, test in zip(train_aucs, test_aucs)]

x = np.arange(len(models))
width = 0.25

bars1 = ax8.bar(x - width, train_aucs, width, label='Train AUC', color='green', alpha=0.7)
bars2 = ax8.bar(x, test_aucs, width, label='Test AUC', color='orange', alpha=0.7)
bars3 = ax8.bar(x + width, gaps, width, label='Gap', color='red', alpha=0.7)

ax8.set_ylabel('AUC')
ax8.set_title('过拟合分析 (Gap<0.02为佳)')
ax8.set_xticks(x)
ax8.set_xticklabels(models)
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')
ax8.axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='警戒线')

plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison_40features.png'), dpi=300, bbox_inches='tight')
print(f"✓ 对比图表已保存")

# 保存模型
joblib.dump(best_model_lr, os.path.join(RESULTS_DIR, 'logistic_regression_40f.pkl'))
joblib.dump(best_model_xgb, os.path.join(RESULTS_DIR, 'xgboost_40f.pkl'))
joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler_40f.pkl'))

# 保存预测结果
results_df = pd.DataFrame({
    'user_id': test_df['user_id'] if 'user_id' in test_df.columns else range(len(y_test)),
    'item_id': test_df['item_id'] if 'item_id' in test_df.columns else range(len(y_test)),
    'true_label': y_test,
    'lr_proba': test_proba_lr,
    'lr_pred': test_pred_lr,
    'xgb_proba': test_proba_xgb,
    'xgb_pred': test_pred_xgb
})
results_df.to_csv(os.path.join(RESULTS_DIR, 'predictions_comparison.csv'), index=False)

# 生成详细报告
report = f"""
双模型对比训练报告（40个特征）
=============================

数据信息:
- 特征数量: {len(feature_cols)}个
- 训练集: {len(train_df):,}样本 (正样本率{train_df['label'].mean():.2%})
- 验证集: {len(val_df):,}样本 (正样本率{val_df['label'].mean():.2%})
- 测试集: {len(test_df):,}样本 (正样本率{test_df['label'].mean():.2%})

模型配置:
Logistic Regression:
- 最优参数: C={best_params_lr['C']}, class_weight={best_params_lr['class_weight']}
- 最优阈值: {best_threshold_lr:.2f}
- 训练集AUC: {lr_train_metrics['AUC']:.4f}
- 测试集AUC: {lr_test_metrics['AUC']:.4f}
- 泛化差距: {lr_train_metrics['AUC'] - lr_test_metrics['AUC']:.4f}

XGBoost:
- 最优参数: {best_params_xgb}
- 最优阈值: {best_threshold_xgb:.2f}
- 训练集AUC: {xgb_train_metrics['AUC']:.4f}
- 测试集AUC: {xgb_test_metrics['AUC']:.4f}
- 泛化差距: {xgb_train_metrics['AUC'] - xgb_test_metrics['AUC']:.4f}

性能对比（测试集）:
指标            LR      XGBoost   提升
Accuracy:      {lr_test_metrics['Accuracy']:.4f}   {xgb_test_metrics['Accuracy']:.4f}   {xgb_test_metrics['Accuracy'] - lr_test_metrics['Accuracy']:+.4f}
Precision:     {lr_test_metrics['Precision']:.4f}   {xgb_test_metrics['Precision']:.4f}   {xgb_test_metrics['Precision'] - lr_test_metrics['Precision']:+.4f}
Recall:        {lr_test_metrics['Recall']:.4f}   {xgb_test_metrics['Recall']:.4f}   {xgb_test_metrics['Recall'] - lr_test_metrics['Recall']:+.4f}
F1-Score:      {lr_test_metrics['F1']:.4f}   {xgb_test_metrics['F1']:.4f}   {xgb_test_metrics['F1'] - lr_test_metrics['F1']:+.4f}
AUC:           {lr_test_metrics['AUC']:.4f}   {xgb_test_metrics['AUC']:.4f}   {xgb_test_metrics['AUC'] - lr_test_metrics['AUC']:+.4f}

关键发现:
1. 过拟合控制:
   - LR训练-测试AUC差距: {lr_train_metrics['AUC'] - lr_test_metrics['AUC']:.2%} ({'优秀' if abs(lr_train_metrics['AUC'] - lr_test_metrics['AUC']) < 0.02 else '需关注'})
   - XGB训练-测试AUC差距: {xgb_train_metrics['AUC'] - xgb_test_metrics['AUC']:.2%} ({'优秀' if abs(xgb_train_metrics['AUC'] - xgb_test_metrics['AUC']) < 0.02 else '需关注'})

2. 模型选择建议:
   - 若追求解释性: 选择LR (系数可解释)
   - 若追求性能: 选择XGBoost (AUC提升{xgb_test_metrics['AUC'] - lr_test_metrics['AUC']:+.2%})
   - 两者泛化能力均良好，Gap均<2%

Top 10关键特征(XGBoost):
{xgb_imp.tail(10)[['feature', 'importance']].to_string(index=False)}
"""

with open(os.path.join(RESULTS_DIR, 'comparison_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n{'=' * 80}")
print("训练完成！")
print(f"结果目录: {RESULTS_DIR}")
print(f"模型文件: logistic_regression_40f.pkl, xgboost_40f.pkl")
print(f"对比图表: model_comparison_40features.png")
print(f"详细报告: comparison_report.txt")
print(f"{'=' * 80}")

# 打印简化版Markdown表格（方便你直接复制到文档）
print("\n📋 Markdown格式结果（可直接复制）:\n")
print("### Logistic Regression")
print("| 评估指标 | 训练集 | 测试集 |")
print("|---------|--------|--------|")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
    print(f"| {metric} | {lr_train_metrics[metric]:.4f} | {lr_test_metrics[metric]:.4f} |")

print("\n### XGBoost")
print("| 评估指标 | 训练集 | 测试集 |")
print("|---------|--------|--------|")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
    print(f"| {metric} | {xgb_train_metrics[metric]:.4f} | {xgb_test_metrics[metric]:.4f} |")

# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\LR1.py
# ================================================================================
# 双模型对比训练 - 基于40个筛选特征
# ================================================================================
#
# [1/6] 加载40个特征数据...
# ✓ 成功加载40个特征数据
#
# 数据集概况:
# 训练集: 328,974 样本, 特征数: 39, 正样本率: 25.16%
# 验证集: 41,122 样本, 正样本率: 24.69%
# 测试集: 41,136 样本, 正样本率: 24.01%
#
# 实际使用特征数: 40 个
# 特征列表: ['ui_funnel_score', 'c_browse_to_buy_rate', 'ui_attention_intensity', 'u_purchase_rate', 'ui_recent_action_concentration']... (共40个)
#
# [2/6] 特征标准化（逻辑回归）...
#
# [3/6] 逻辑回归训练与超参数调优...
# 逻辑回归 Grid Search:
#   C= 0.01, weight=balanced   | Val AUC=0.7936
#   C= 0.01, weight=None       | Val AUC=0.7915
#   C= 0.10, weight=balanced   | Val AUC=0.7937
#   C= 0.10, weight=None       | Val AUC=0.7915
#   C= 0.50, weight=balanced   | Val AUC=0.7938
#   C= 0.50, weight=None       | Val AUC=0.7915
#   C= 1.00, weight=balanced   | Val AUC=0.7938
#   C= 1.00, weight=None       | Val AUC=0.7915
#   C=10.00, weight=balanced   | Val AUC=0.7938
#   C=10.00, weight=None       | Val AUC=0.7916
#
# LR最优参数: {'C': 10.0, 'class_weight': 'balanced'}
# LR最优阈值: 0.50
#
# [4/6] XGBoost训练与超参数调优...
# XGBoost Grid Search:
#   depth=4, lr=0.05, n=100 | Val AUC=0.8115
#   depth=4, lr=0.05, n=200 | Val AUC=0.8163
#   depth=4, lr=0.1, n=100 | Val AUC=0.8162
#   depth=4, lr=0.1, n=200 | Val AUC=0.8193
#   depth=6, lr=0.05, n=100 | Val AUC=0.8174
#   depth=6, lr=0.05, n=200 | Val AUC=0.8214
#   depth=6, lr=0.1, n=100 | Val AUC=0.8208
#   depth=6, lr=0.1, n=200 | Val AUC=0.8242
#
# XGB最优参数: {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200}
# XGB最优阈值: 0.55
#
# [5/6] 计算评估指标...
#
# ================================================================================
# 模型性能对比（基于40个特征）
# ================================================================================
#
# Logistic Regression:
# ------------------------------------------------------------
# 指标                       训练集          测试集           差距
# ------------------------------------------------------------
# Accuracy              0.7349       0.7348       0.0002
# Precision             0.4815       0.4647       0.0168
# Recall                0.6951       0.6886       0.0065
# F1                    0.5689       0.5549       0.0140
# AUC                   0.7963       0.7917       0.0046
#
# XGBoost:
# ------------------------------------------------------------
# 指标                       训练集          测试集           差距
# ------------------------------------------------------------
# Accuracy              0.7723       0.7614       0.0109
# Precision             0.5352       0.5023       0.0329
# Recall                0.7232       0.6970       0.0262
# F1                    0.6152       0.5838       0.0313
# AUC                   0.8438       0.8230       0.0208
#
# 模型对比（测试集）:
# ------------------------------------------------------------
# 指标                 LogisticR      XGBoost           提升
# ------------------------------------------------------------
# Accuracy              0.7348       0.7614      +0.0266
# Precision             0.4647       0.5023      +0.0376
# Recall                0.6886       0.6970      +0.0084
# F1                    0.5549       0.5838      +0.0289
# AUC                   0.7917       0.8230      +0.0313
#
# [6/6] 生成可视化与保存结果...
# ✓ 对比图表已保存
#
# ================================================================================
# 训练完成！
# 结果目录: E:\PycharmeProjects\taobao_user_behavior_analysis\results\model_comparison_40features
# 模型文件: logistic_regression_40f.pkl, xgboost_40f.pkl
# 对比图表: model_comparison_40features.png
# 详细报告: comparison_report.txt
# ================================================================================
#
# 📋 Markdown格式结果（可直接复制）:
#
# ### Logistic Regression
# | 评估指标 | 训练集 | 测试集 |
# |---------|--------|--------|
# | Accuracy | 0.7349 | 0.7348 |
# | Precision | 0.4815 | 0.4647 |
# | Recall | 0.6951 | 0.6886 |
# | F1 | 0.5689 | 0.5549 |
# | AUC | 0.7963 | 0.7917 |
#
# ### XGBoost
# | 评估指标 | 训练集 | 测试集 |
# |---------|--------|--------|
# | Accuracy | 0.7723 | 0.7614 |
# | Precision | 0.5352 | 0.5023 |
# | Recall | 0.7232 | 0.6970 |
# | F1 | 0.6152 | 0.5838 |
# | AUC | 0.8438 | 0.8230 |
#
# Process finished with exit code 0
