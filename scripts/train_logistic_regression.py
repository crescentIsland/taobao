"""
逻辑回归训练脚本 - 使用XGBoost筛选后的特征
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt

# 路径配置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(project_root, 'data', 'features_for_lr')


def load_features():
    """加载保存的特征数据"""
    print("Loading features from:", FEATURE_DIR)

    # 加载数据
    train_df = pd.read_parquet(os.path.join(FEATURE_DIR, 'train_features.parquet'))
    val_df = pd.read_parquet(os.path.join(FEATURE_DIR, 'val_features.parquet'))
    test_df = pd.read_parquet(os.path.join(FEATURE_DIR, 'test_features.parquet'))

    # 加载特征信息
    with open(os.path.join(FEATURE_DIR, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)

    feature_names = feature_info['selected_features']

    # 分离X和y
    X_train = train_df[feature_names]
    y_train = train_df['label']
    X_val = val_df[feature_names]
    y_val = val_df['label']
    X_test = test_df[feature_names]
    y_test = test_df['label']

    print(f"Loaded {len(feature_names)} features")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """训练逻辑回归并评估"""

    # 标准化（逻辑回归必需）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 训练逻辑回归（调参版本）
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',  # 处理不平衡
        C=0.1,  # 正则化强度（可尝试0.01, 0.1, 1, 10）
        penalty='l2',
        solver='lbfgs',
        random_state=42
    )
    lr.fit(X_train_s, y_train)

    # 评估
    results = {}
    for name, X, y in [('Train', X_train_s, y_train),
                       ('Validation', X_val_s, y_val),
                       ('Test', X_test_s, y_test)]:
        prob = lr.predict_proba(X)[:, 1]
        pred = lr.predict(X)
        auc = roc_auc_score(y, prob)
        results[name] = {'auc': auc, 'prob': prob, 'pred': pred}

        print(f"\n{name} AUC: {auc:.4f}")
        print(classification_report(y, pred, target_names=['No Buy', 'Buy'], digits=4))

    # 特征系数分析（可解释性）
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': lr.coef_[0],
        'abs_coef': np.abs(lr.coef_[0]),
        'odds_ratio': np.exp(lr.coef_[0])  # 优势比，>1表示促进购买
    }).sort_values('abs_coef', ascending=False)

    print("\n" + "=" * 60)
    print("Logistic Regression Coefficients (Top 20):")
    print(coef_df.head(20).to_string(index=False))
    print("=" * 60)

    # 保存系数
    coef_df.to_csv(os.path.join(FEATURE_DIR, 'logistic_regression_coefficients.csv'), index=False)

    # 绘制特征重要性对比（XGBoost vs LR）
    try:
        xgb_imp = pd.read_csv(os.path.join(FEATURE_DIR, 'xgboost_feature_importance.csv'))
        compare_importance(xgb_imp, coef_df)
    except:
        pass

    return lr, scaler, coef_df


def compare_importance(xgb_df, lr_df):
    """对比XGBoost和LR的特征重要性"""
    merged = xgb_df.merge(lr_df[['feature', 'abs_coef']], on='feature')
    merged.columns = ['feature', 'xgb_importance', 'lr_importance']

    # 归一化到0-1范围便于对比
    merged['xgb_norm'] = merged['xgb_importance'] / merged['xgb_importance'].max()
    merged['lr_norm'] = merged['lr_importance'] / merged['lr_importance'].max()

    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(merged))

    plt.barh(x_pos - 0.2, merged['xgb_norm'], 0.4, label='XGBoost', alpha=0.8)
    plt.barh(x_pos + 0.2, merged['lr_norm'], 0.4, label='Logistic Regression', alpha=0.8)
    plt.yticks(x_pos, merged['feature'], fontsize=8)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance Comparison: XGBoost vs Logistic Regression')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_DIR, 'feature_importance_comparison.png'), dpi=150)
    print(f"\nComparison plot saved to: {FEATURE_DIR}/feature_importance_comparison.png")


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_features()
    model, scaler, coefficients = train_logistic_regression(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )

# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\train_logistic_regression.py
# Loading features from: E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr
# Loaded 40 features
# Train: 328974, Val: 41122, Test: 41136
#
# Training Logistic Regression...
#
# Train AUC: 0.7963
#               precision    recall  f1-score   support
#
#       No Buy     0.8795    0.7482    0.8085    246195
#          Buy     0.4813    0.6950    0.5688     82779
#
#     accuracy                         0.7348    328974
#    macro avg     0.6804    0.7216    0.6887    328974
# weighted avg     0.7793    0.7348    0.7482    328974
#
#
# Validation AUC: 0.7937
#               precision    recall  f1-score   support
#
#       No Buy     0.8812    0.7466    0.8084     30971
#          Buy     0.4727    0.6930    0.5620     10151
#
#     accuracy                         0.7334     41122
#    macro avg     0.6770    0.7198    0.6852     41122
# weighted avg     0.7804    0.7334    0.7476     41122
#
#
# Test AUC: 0.7917
#               precision    recall  f1-score   support
#
#       No Buy     0.8839    0.7496    0.8112     31258
#          Buy     0.4649    0.6884    0.5550      9878
#
#     accuracy                         0.7349     41136
#    macro avg     0.6744    0.7190    0.6831     41136
# weighted avg     0.7833    0.7349    0.7497     41136
#
#
# ============================================================
# Logistic Regression Coefficients (Top 20):
#                   feature      coef  abs_coef  odds_ratio
#           ui_funnel_score  1.460480  1.460480    4.308027
#          ui_total_actions -1.452061  1.452061    0.234087
#           uc_browse_count -0.645871  0.645871    0.524206
#          uc_total_actions  0.524726  0.524726    1.689996
#              avg_interval -0.483795  0.483795    0.616440
#            c_browse_count -0.414720  0.414720    0.660526
#        i_cart_to_buy_rate  0.373631  0.373631    1.453002
#      c_browse_to_buy_rate  0.346400  0.346400    1.413968
#              c_cart_count  0.320171  0.320171    1.377364
#         c_fav_to_buy_rate  0.246584  0.246584    1.279647
#         ui_view_frequency  0.233005  0.233005    1.262388
#           u_purchase_rate  0.218655  0.218655    1.244402
#         i_fav_to_buy_rate -0.206935  0.206935    0.813072
#       ui_last_action_hour  0.180047  0.180047    1.197274
#       u_total_actions_21d -0.175392  0.175392    0.839128
#          u_purchase_count  0.168271  0.168271    1.183258
#         c_popularity_rank  0.167123  0.167123    1.181900
# ui_days_since_last_action -0.165251  0.165251    0.847681
#        c_cart_to_buy_rate  0.160148  0.160148    1.173684
#              i_cart_count  0.143048  0.143048    1.153785
# ============================================================
#
# Comparison plot saved to: E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr/feature_importance_comparison.png
#
# Process finished with exit code 0
