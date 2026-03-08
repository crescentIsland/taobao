"""
逻辑回归模型训练与验证（清洗版）
基于筛选后的28个特征（已删除泄露特征），验证用户是否购买（二分类）
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

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, accuracy_score, precision_recall_curve)
from sklearn.model_selection import GridSearchCV

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径配置 ====================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(project_root, 'results', 'logistic_regression_final')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 修改：使用清洗后的特征文件（已删除泄露特征）
FEATURES_DIR = os.path.join(project_root, 'results', 'full_features_selected')

print("=" * 80)
print("逻辑回归训练 - 购买行为验证（清洗后28特征版）")
print("=" * 80)

# ==================== 1. 加载数据 ====================
print("\n[1/5] 加载特征数据...")

# 修改：读取清洗后的文件（删除了ui_has_purchased和ui_purchase_count）
train_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'train_selected_clean.parquet'))
val_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'val_selected_clean.parquet'))
test_df = pd.read_parquet(os.path.join(FEATURES_DIR, 'test_selected_clean.parquet'))

# 安全检查：确保泄露特征已被删除（防御性编程）
leakage_features = ['ui_has_purchased', 'ui_purchase_count']
for df, name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
    leaked = [f for f in leakage_features if f in df.columns]
    if leaked:
        print(f"[!] {name}发现泄露特征 {leaked}，执行自动删除")
        df.drop(columns=leaked, inplace=True)

print(f"训练集: {len(train_df):,} 样本, 正样本率: {train_df['label'].mean():.2%}")
print(f"验证集: {len(val_df):,} 样本, 正样本率: {val_df['label'].mean():.2%}")
print(f"测试集: {len(test_df):,} 样本, 正样本率: {test_df['label'].mean():.2%}")

# 分离特征和标签（排除ID列）
feature_cols = [c for c in train_df.columns if c not in ['user_id', 'item_id', 'label']]
print(f"特征数: {len(feature_cols)} 个（已清洗泄露特征）")

X_train, y_train = train_df[feature_cols], train_df['label']
X_val, y_val = val_df[feature_cols], val_df['label']
X_test, y_test = test_df[feature_cols], test_df['label']

# ==================== 2. 特征标准化 ====================
print("\n[2/5] 特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 保存scaler
joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler.pkl'))

# ==================== 3. 超参数调优（基于验证集） ====================
print("\n[3/5] 超参数调优（C值 + class_weight）...")

# 定义参数网格
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'class_weight': ['balanced', None]
}

best_auc = 0
best_params = {}
best_model = None

print("Grid Search 进度:")
for C in param_grid['C']:
    for cw in param_grid['class_weight']:
        model = LogisticRegression(
            C=C,
            class_weight=cw,
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        # 在验证集上评估
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)

        print(f"  C={C:6.2f}, class_weight={str(cw):10s} | Val AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_params = {'C': C, 'class_weight': cw}
            best_model = model

print(f"\n最优参数: {best_params}")
print(f"验证集最佳AUC: {best_auc:.4f}")

# ==================== 4. 阈值调优（解决训练集/验证集分布不一致） ====================
print("\n[4/5] 阈值调优...")

# 在验证集上寻找最佳阈值（F1最优）
val_proba = best_model.predict_proba(X_val_scaled)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_threshold = 0.5

print("阈值搜索:")
for thresh in thresholds:
    pred = (val_proba >= thresh).astype(int)
    f1 = f1_score(y_val, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
    print(f"  Threshold={thresh:.2f} | F1={f1:.4f}")

print(f"\n最优阈值: {best_threshold:.2f} (F1={best_f1:.4f})")
print(f"注意：训练集正样本率25%，验证集9%，默认0.5阈值偏高，调整至{best_threshold:.2f}")

# 保存阈值
with open(os.path.join(RESULTS_DIR, 'best_threshold.txt'), 'w') as f:
    f.write(f"{best_threshold}")

# ==================== 5. 测试集最终评估 ====================
print("\n[5/5] 测试集最终评估...")

test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

# 基础指标
test_auc = roc_auc_score(y_test, test_proba)
test_f1 = f1_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\n{'=' * 60}")
print("测试集性能报告（清洗后特征）")
print(f"{'=' * 60}")
print(f"AUC-ROC:  {test_auc:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print(f"Precision:{test_precision:.4f}")
print(f"Recall:   {test_recall:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"{'=' * 60}")

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, test_pred, target_names=['未购买', '购买']))

# 混淆矩阵
cm = confusion_matrix(y_test, test_pred)
print(f"\n混淆矩阵:")
print(f"          预测未购买  预测购买")
print(f"实际未购买 {cm[0, 0]:8,}  {cm[0, 1]:8,}  ( specificity={cm[0, 0] / (cm[0, 0] + cm[0, 1]):.2%} )")
print(f"实际购买   {cm[1, 0]:8,}  {cm[1, 1]:8,}  ( recall={cm[1, 1] / (cm[1, 0] + cm[1, 1]):.2%} )")

# ==================== 6. 可视化与保存 ====================
print("\n[6/6] 生成可视化与保存模型...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. ROC曲线
fpr_val, tpr_val, _ = roc_curve(y_val, val_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, test_proba)

axes[0, 0].plot(fpr_val, tpr_val, label=f'验证集 (AUC={best_auc:.4f})', linewidth=2, linestyle='--')
axes[0, 0].plot(fpr_test, tpr_test, label=f'测试集 (AUC={test_auc:.4f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='随机基线 (AUC=0.5)')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC曲线对比')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Precision-Recall曲线
precision_val, recall_val, _ = precision_recall_curve(y_val, val_proba)
precision_test, recall_test, _ = precision_recall_curve(y_test, test_proba)

axes[0, 1].plot(recall_val, precision_val, label='验证集', linewidth=2, linestyle='--')
axes[0, 1].plot(recall_test, precision_test, label='测试集', linewidth=2)
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall曲线')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 预测概率分布
axes[1, 0].hist(test_proba[y_test == 0], bins=50, alpha=0.5, label='未购买', density=True, color='red')
axes[1, 0].hist(test_proba[y_test == 1], bins=50, alpha=0.5, label='购买', density=True, color='green')
axes[1, 0].axvline(best_threshold, color='black', linestyle='--', label=f'阈值={best_threshold:.2f}')
axes[1, 0].set_xlabel('预测购买概率')
axes[1, 0].set_ylabel('密度')
axes[1, 0].set_title('测试集预测概率分布')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 特征重要性（逻辑回归系数）
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coef': best_model.coef_[0],
    'abs_coef': np.abs(best_model.coef_[0])
}).sort_values('abs_coef', ascending=True).tail(15)

colors = ['red' if c < 0 else 'green' for c in coef_df['coef']]
axes[1, 1].barh(coef_df['feature'], coef_df['coef'], color=colors, alpha=0.7)
axes[1, 1].set_xlabel('回归系数 (负=抑制购买, 正=促进购买)')
axes[1, 1].set_title('Top 15 特征影响力')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_evaluation.png'), dpi=300, bbox_inches='tight')
print(f"✓ 可视化已保存: {os.path.join(RESULTS_DIR, 'model_evaluation.png')}")

# 保存模型和结果
joblib.dump(best_model, os.path.join(RESULTS_DIR, 'logistic_regression_model.pkl'))

# 保存预测结果
test_results = pd.DataFrame({
    'user_id': test_df['user_id'],
    'item_id': test_df['item_id'],
    'true_label': y_test,
    'pred_proba': test_proba,
    'pred_label': test_pred
})
test_results.to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False)

# 保存特征列表（清洗后的）
with open(os.path.join(RESULTS_DIR, 'final_features_28.txt'), 'w', encoding='utf-8') as f:
    f.write("# 最终使用的28个特征（已删除泄露特征）\n\n")
    for i, feat in enumerate(feature_cols, 1):
        f.write(f"{i}. {feat}\n")

# 保存训练报告
report = f"""
逻辑回归训练报告（数据清洗版）
=============================

数据说明:
- 已删除泄露特征: ui_has_purchased, ui_purchase_count
- 实际使用特征数: {len(feature_cols)}个

数据划分:
- 训练集: {len(train_df):,} 样本 (正样本率: {train_df['label'].mean():.2%})
- 验证集: {len(val_df):,} 样本 (正样本率: {val_df['label'].mean():.2%})
- 测试集: {len(test_df):,} 样本 (正样本率: {test_df['label'].mean():.2%})

最优参数:
- C (正则化强度): {best_params['C']}
- class_weight: {best_params['class_weight']}
- 决策阈值: {best_threshold:.2f}

测试集性能:
- AUC-ROC: {test_auc:.4f}
- F1-Score: {test_f1:.4f}
- Precision: {test_precision:.4f}
- Recall: {test_recall:.4f}
- Accuracy: {test_accuracy:.4f}

Top 5 重要特征 (促进购买):
{coef_df.tail(5)[['feature', 'coef']].to_string(index=False)}

Top 5 重要特征 (抑制购买):
{coef_df.head(5)[['feature', 'coef']].to_string(index=False)}

重要说明:
本次训练使用了清洗后的28个特征（原30个特征删除了2个泄露特征）。
这确保了模型泛化能力的真实性，避免了"用购买行为预测购买"的数据泄露问题。
"""

with open(os.path.join(RESULTS_DIR, 'training_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n{'=' * 80}")
print("训练完成！（已使用清洗后28个特征）")
print(f"模型保存: {os.path.join(RESULTS_DIR, 'logistic_regression_model.pkl')}")
print(f"预测结果: {os.path.join(RESULTS_DIR, 'test_predictions.csv')}")
print(f"特征列表: {os.path.join(RESULTS_DIR, 'final_features_28.txt')}")
print(f"训练报告: {os.path.join(RESULTS_DIR, 'training_report.txt')}")
print(f"{'=' * 80}")

# 输出关键发现
print("\n关键发现:")
if test_auc > 0.75:
    print(f"✓ 模型表现优秀 (AUC={test_auc:.4f} > 0.75)")
elif test_auc > 0.6:
    print(f"△ 模型表现尚可 (AUC={test_auc:.4f})，但仍有提升空间")
else:
    print(f"✗ 模型表现较差 (AUC={test_auc:.4f})，建议检查特征或数据")

print(f"\n数据泄露检查:")
print(f"  ✓ 已确认删除: {leakage_features}")
print(f"  ✓ 当前特征数: {len(feature_cols)}个（干净特征）")

print(f"\n阈值调整说明:")
print(f"  - 训练集正样本多(25%)，模型倾向预测高概率")
print(f"  - 验证集正样本少(9%)，默认阈值0.5会导致Precision下降")
print(f"  - 调整至{best_threshold:.2f}后，F1从{f1_score(y_test, (test_proba >= 0.5).astype(int)):.4f}提升至{test_f1:.4f}")