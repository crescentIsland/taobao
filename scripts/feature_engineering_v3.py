"""
淘宝用户行为预测 - 高性能完整版（单进程+修复类型错误）
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import RFECV
import xgboost as xgb
from tqdm import tqdm  # 进度条
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. 路径配置 ====================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
DATA_PATH = os.path.join(project_root, 'data', 'processed', 'user_action_processed.parquet')


# ==================== 2. 数据加载（修复：保持datetime类型） ====================
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    # 关键修复：date列保持为datetime，不转date对象（避免.dt报错）
    df['date'] = pd.to_datetime(df['date'])
    return df


# ==================== 3. 预计算引擎 ====================
class PrecomputedStats:
    def __init__(self, df):
        print("Precomputing daily statistics...")
        self.df = df.copy()

        print("  Computing item daily stats...")
        self.item_daily = self._compute_item_daily()

        print("  Computing user daily stats...")
        self.user_daily = self._compute_user_daily()

        print("  Computing category daily stats...")
        self.cat_daily = self._compute_category_daily()

        print("  Computing UI daily stats...")
        self.ui_daily = self._compute_ui_daily()

        print("  Computing UC daily stats...")
        self.uc_daily = self._compute_uc_daily()

        print("Precomputation completed!")

    def _compute_item_daily(self):
        stats = self.df.groupby(['item_id', 'date'])['behavior_type'].agg([
            ('browse', lambda x: (x == '1').sum()),
            ('fav', lambda x: (x == '2').sum()),
            ('cart', lambda x: (x == '3').sum()),
            ('purchase', lambda x: (x == '4').sum())
        ]).reset_index()
        return stats

    def _compute_user_daily(self):
        stats = self.df.groupby(['user_id', 'date'])['behavior_type'].agg([
            ('total', 'count'),
            ('purchase', lambda x: (x == '4').sum())
        ]).reset_index()

        # 最活跃小时（众数）
        hour_mode = self.df.groupby(['user_id', 'date'])['hour'].apply(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 12
        ).reset_index(name='hour_mode')

        return stats.merge(hour_mode, on=['user_id', 'date'])

    def _compute_category_daily(self):
        stats = self.df.groupby(['item_category', 'date'])['behavior_type'].agg([
            ('browse', lambda x: (x == '1').sum()),
            ('fav', lambda x: (x == '2').sum()),
            ('cart', lambda x: (x == '3').sum()),
            ('purchase', lambda x: (x == '4').sum())
        ]).reset_index()
        return stats

    def _compute_ui_daily(self):
        stats = self.df.groupby(['user_id', 'item_id', 'date'])['behavior_type'].agg([
            ('total', 'count'),
            ('view', lambda x: (x == '1').sum()),
            ('fav', lambda x: (x == '2').sum()),
            ('cart', lambda x: (x == '3').sum()),
            ('purchase', lambda x: (x == '4').sum())
        ]).reset_index()
        return stats

    def _compute_uc_daily(self):
        stats = self.df.groupby(['user_id', 'item_category', 'date'])['behavior_type'].agg([
            ('total', 'count'),
            ('purchase', lambda x: (x == '4').sum()),
            ('fav', lambda x: (x == '2').sum()),
            ('cart', lambda x: (x == '3').sum()),
            ('browse', lambda x: (x == '1').sum())
        ]).reset_index()
        return stats


# ==================== 4. 快速特征工程（修复：确保timestamp类型） ====================
class FastFeatureEngineer:
    def __init__(self, precomputed, predict_date):
        self.pc = precomputed
        # 关键修复：确保是pd.Timestamp类型
        self.date = pd.Timestamp(predict_date)
        self.hist_dates = [self.date - timedelta(days=i) for i in range(1, 22)]

    def generate_features(self, samples):
        result = samples.copy()
        result = self._add_item_features(result)
        result = self._add_user_features(result)
        result = self._add_ui_features(result)
        result = self._add_category_features(result)
        result = self._add_uc_features(result)
        return result

    def _add_item_features(self, df):
        hist = self.pc.item_daily[self.pc.item_daily['date'].isin(self.hist_dates)]

        item_hist = hist.groupby('item_id').agg({
            'browse': 'sum',
            'fav': 'sum',
            'cart': 'sum',
            'purchase': 'sum',
            'date': 'max'
        }).reset_index()

        item_hist.columns = ['item_id', 'i_browse_count', 'i_fav_count', 'i_cart_count',
                             'i_purchase_count', 'last_date']

        item_hist['i_browse_to_buy_rate'] = (item_hist['i_purchase_count'] + 1) / (item_hist['i_browse_count'] + 10)
        item_hist['i_cart_to_buy_rate'] = (item_hist['i_purchase_count'] + 1) / (item_hist['i_cart_count'] + 10)
        item_hist['i_fav_to_buy_rate'] = (item_hist['i_purchase_count'] + 1) / (item_hist['i_fav_count'] + 10)

        # 修复：确保日期相减得到timedelta
        item_hist['i_days_since_last_action'] = (self.date - item_hist['last_date']).dt.days

        for window in [1, 3, 7, 15, 21]:
            window_dates = self.hist_dates[:window]
            window_data = hist[hist['date'].isin(window_dates)]
            if len(window_data) > 0:
                window_agg = window_data.groupby('item_id')[['browse', 'fav', 'cart', 'purchase']].sum().sum(
                    axis=1).reset_index()
                window_agg.columns = ['item_id', f'i_act_total_{window}d']
                item_hist = item_hist.merge(window_agg, on='item_id', how='left')
            else:
                item_hist[f'i_act_total_{window}d'] = 0

        result = df.merge(item_hist.drop('last_date', axis=1), on='item_id', how='left')
        return result.fillna(0)

    def _add_user_features(self, df):
        hist = self.pc.user_daily[self.pc.user_daily['date'].isin(self.hist_dates)]

        user_hist = hist.groupby('user_id').agg({
            'total': 'sum',
            'purchase': 'sum'
        }).reset_index()
        user_hist.columns = ['user_id', 'u_total_actions', 'u_purchase_count']
        user_hist['u_purchase_rate'] = user_hist['u_purchase_count'] / (user_hist['u_total_actions'] + 1)

        for window in [1, 3, 7, 15, 21]:
            window_dates = self.hist_dates[:window]
            window_data = hist[hist['date'].isin(window_dates)]
            if len(window_data) > 0:
                window_agg = window_data.groupby('user_id')['total'].sum().reset_index()
                window_agg.columns = ['user_id', f'u_total_actions_{window}d']
                user_hist = user_hist.merge(window_agg, on='user_id', how='left')
            else:
                user_hist[f'u_total_actions_{window}d'] = 0

        # 最活跃小时
        latest_hour = hist.loc[hist.groupby('user_id')['date'].idxmax()][['user_id', 'hour_mode']]
        latest_hour.columns = ['user_id', 'u_preferred_hour']
        user_hist = user_hist.merge(latest_hour, on='user_id', how='left')

        result = df.merge(user_hist, on='user_id', how='left')
        return result.fillna(0)

    def _add_ui_features(self, df):
        hist = self.pc.ui_daily[self.pc.ui_daily['date'].isin(self.hist_dates)]

        if len(hist) == 0:
            # 如果没有历史UI数据，填充0
            for col in ['ui_total_actions', 'ui_view_count', 'ui_fav_count', 'ui_cart_count',
                        'ui_has_carted', 'ui_has_faved', 'ui_days_since_last_action',
                        'ui_funnel_score', 'ui_time_from_first_view', 'ui_view_frequency',
                        'ui_is_impulse_buy_candidate', 'ui_has_fav_cart_pattern']:
                df[col] = 0
            return df

        ui_hist = hist.groupby(['user_id', 'item_id']).agg({
            'total': 'sum',
            'view': 'sum',
            'fav': 'sum',
            'cart': 'sum',
            'purchase': 'sum',
            'date': ['min', 'max']
        }).reset_index()

        ui_hist.columns = ['user_id', 'item_id', 'ui_total_actions', 'ui_view_count',
                           'ui_fav_count', 'ui_cart_count', 'ui_purchase_count_hist',
                           'first_date', 'last_date']

        ui_hist['ui_has_carted'] = (ui_hist['ui_cart_count'] > 0).astype(int)
        ui_hist['ui_has_faved'] = (ui_hist['ui_fav_count'] > 0).astype(int)
        ui_hist['ui_days_since_last_action'] = (self.date - ui_hist['last_date']).dt.days
        ui_hist['ui_time_from_first_view'] = (self.date - ui_hist['first_date']).dt.total_seconds() / 3600

        ui_hist['ui_funnel_score'] = (
                ui_hist['ui_view_count'] * 0.1 +
                ui_hist['ui_fav_count'] * 0.2 +
                ui_hist['ui_cart_count'] * 0.3
        )

        view_days = hist[hist['view'] > 0].groupby(['user_id', 'item_id'])['date'].nunique().reset_index(
            name='ui_view_days')
        ui_hist = ui_hist.merge(view_days, on=['user_id', 'item_id'], how='left')
        ui_hist['ui_view_days'] = ui_hist['ui_view_days'].fillna(1).replace(0, 1)
        ui_hist['ui_view_frequency'] = ui_hist['ui_view_count'] / ui_hist['ui_view_days']

        ui_hist['ui_is_impulse_buy_candidate'] = (
                (ui_hist['ui_view_count'] <= 2) & (ui_hist['ui_has_carted'] == 1)
        ).astype(int)

        ui_hist['ui_has_fav_cart_pattern'] = (
                (ui_hist['ui_has_faved'] == 1) & (ui_hist['ui_has_carted'] == 1)
        ).astype(int)

        result = df.merge(
            ui_hist.drop(['first_date', 'last_date'], axis=1),
            on=['user_id', 'item_id'],
            how='left'
        )
        return result.fillna(0)

    def _add_category_features(self, df):
        hist = self.pc.cat_daily[self.pc.cat_daily['date'].isin(self.hist_dates)]

        cat_hist = hist.groupby('item_category').agg({
            'browse': 'sum',
            'fav': 'sum',
            'cart': 'sum',
            'purchase': 'sum'
        }).reset_index()

        cat_hist.columns = ['item_category', 'c_browse_count', 'c_fav_count', 'c_cart_count', 'c_purchase_count']

        cat_hist['c_browse_to_buy_rate'] = (cat_hist['c_purchase_count'] + 1) / (cat_hist['c_browse_count'] + 10)
        cat_hist['c_cart_to_buy_rate'] = (cat_hist['c_purchase_count'] + 1) / (cat_hist['c_cart_count'] + 10)

        for window in [7, 15, 21]:
            window_dates = self.hist_dates[:window]
            window_data = hist[hist['date'].isin(window_dates)]
            if len(window_data) > 0:
                window_agg = window_data.groupby('item_category')[['browse', 'fav', 'cart', 'purchase']].sum().sum(
                    axis=1).reset_index()
                window_agg.columns = ['item_category', f'c_act_total_{window}d']
                cat_hist = cat_hist.merge(window_agg, on='item_category', how='left')
            else:
                cat_hist[f'c_act_total_{window}d'] = 0

        cat_hist['c_popularity_rank'] = cat_hist['c_act_total_7d'].rank(ascending=False, method='dense')

        result = df.merge(cat_hist, on='item_category', how='left')
        return result.fillna(0)

    def _add_uc_features(self, df):
        hist = self.pc.uc_daily[self.pc.uc_daily['date'].isin(self.hist_dates)]

        if len(hist) == 0:
            for col in ['uc_total_actions', 'uc_purchase_count', 'uc_fav_count',
                        'uc_cart_count', 'uc_browse_count', 'uc_preference_score',
                        'uc_recency', 'uc_category_share']:
                df[col] = 0
            return df

        uc_hist = hist.groupby(['user_id', 'item_category']).agg({
            'total': 'sum',
            'purchase': 'sum',
            'fav': 'sum',
            'cart': 'sum',
            'browse': 'sum',
            'date': 'max'
        }).reset_index()

        uc_hist.columns = ['user_id', 'item_category', 'uc_total_actions', 'uc_purchase_count',
                           'uc_fav_count', 'uc_cart_count', 'uc_browse_count', 'last_date']

        uc_hist['uc_preference_score'] = (
                                                 uc_hist['uc_purchase_count'] * 0.5 +
                                                 uc_hist['uc_cart_count'] * 0.3 +
                                                 uc_hist['uc_fav_count'] * 0.2
                                         ) / (uc_hist['uc_total_actions'] + 1)

        uc_hist['uc_recency'] = (self.date - uc_hist['last_date']).dt.days

        user_total = hist.groupby('user_id')['total'].sum().reset_index()
        user_total.columns = ['user_id', 'u_total_all']
        uc_hist = uc_hist.merge(user_total, on='user_id', how='left')
        uc_hist['uc_category_share'] = uc_hist['uc_total_actions'] / (uc_hist['u_total_all'] + 1)

        result = df.merge(uc_hist.drop('last_date', axis=1), on=['user_id', 'item_category'], how='left')
        return result.fillna(0)


# ==================== 5. 样本生成与划分 ====================
def generate_daily_samples(df, date, neg_ratio=3):
    """生成某一天的正负样本"""
    # 关键修复：确保date是datetime类型用于比较
    if isinstance(date, str):
        date = pd.Timestamp(date).date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()

    day_df = df[df['date'].dt.date == date]
    if len(day_df) == 0:
        return pd.DataFrame()

    positive = day_df[day_df['behavior_type'] == '4'][['user_id', 'item_id', 'item_category']].drop_duplicates()
    if len(positive) == 0:
        return pd.DataFrame()
    positive['label'] = 1

    interacted = day_df[day_df['behavior_type'].isin(['1', '2', '3'])][
        ['user_id', 'item_id', 'item_category']].drop_duplicates()

    negative = interacted.merge(
        positive[['user_id', 'item_id']],
        on=['user_id', 'item_id'],
        how='left',
        indicator=True
    )
    negative = negative[negative['_merge'] == 'left_only'][['user_id', 'item_id', 'item_category']]

    if len(negative) > len(positive) * neg_ratio:
        negative = negative.sample(n=len(positive) * neg_ratio, random_state=42)

    negative['label'] = 0
    samples = pd.concat([positive, negative], ignore_index=True)
    samples['predict_date'] = date

    return samples


def split_daily_8_1_1(samples_df):
    """按日分层8:1:1"""
    train_list, val_list, test_list = [], [], []
    dates = sorted(samples_df['predict_date'].unique())

    for date in dates:
        day_data = samples_df[samples_df['predict_date'] == date].sample(frac=1, random_state=42)
        n = len(day_data)
        if n < 10:
            train_list.append(day_data)
            continue

        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_list.append(day_data.iloc[:train_end])
        val_list.append(day_data.iloc[train_end:val_end])
        test_list.append(day_data.iloc[val_end:])

    train = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    return train, val, test


# ==================== 6. 特征筛选 ====================
class FeatureSelector:
    def __init__(self, max_features=30):
        self.max_features = max_features
        self.selected_features = None
        self.scaler = StandardScaler()

    def fit(self, train_df, target_col='label'):
        exclude_cols = ['user_id', 'item_id', 'item_category', 'label', 'predict_date']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols
                        and pd.api.types.is_numeric_dtype(train_df[c])]

        X = train_df[feature_cols].fillna(0)
        y = train_df[target_col]

        # 1. 低方差
        variances = X.var()
        valid_feats = variances[variances > 0.001].index.tolist()
        if len(valid_feats) < 10:
            valid_feats = feature_cols  # 如果过滤后太少，保留全部
        X = X[valid_feats]
        print(f"After variance filter: {len(valid_feats)}")

        # 2. IV筛选
        print("Calculating IV...")
        iv_scores = {}
        for col in valid_feats:
            try:
                x_clean = X[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                if x_clean.nunique() < 2:
                    continue
                bins = pd.qcut(x_clean, q=5, duplicates='drop')
                iv_table = pd.crosstab(bins, y)
                if iv_table.shape[0] < 2 or iv_table.shape[1] < 2:
                    continue
                iv_table = iv_table / iv_table.sum()
                iv_table['woe'] = np.log((iv_table[0] + 0.0001) / (iv_table[1] + 0.0001))
                iv_table['iv'] = (iv_table[0] - iv_table[1]) * iv_table['woe']
                iv_scores[col] = iv_table['iv'].sum()
            except:
                iv_scores[col] = 0

        iv_df = pd.DataFrame(list(iv_scores.items()), columns=['feature', 'iv'])
        iv_df = iv_df.sort_values('iv', ascending=False)
        iv_selected = iv_df[iv_df['iv'] > 0.02]['feature'].tolist()
        if len(iv_selected) > 40:
            iv_selected = iv_df.head(40)['feature'].tolist()
        if len(iv_selected) < 5:
            iv_selected = valid_feats[:min(20, len(valid_feats))]
        print(f"After IV filter: {len(iv_selected)}")

        # 3. XGBoost
        print("XGBoost selection...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'
        )
        xgb_model.fit(X[iv_selected], y)

        importance = pd.DataFrame({
            'feature': iv_selected,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        top_35 = importance.head(35)['feature'].tolist()

        # 4. RFECV
        print("RFECV...")
        X_scaled = self.scaler.fit_transform(X[top_35])

        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        rfecv = RFECV(
            estimator=lr, step=1, cv=3,
            scoring='roc_auc',
            min_features_to_select=min(self.max_features, len(top_35)),
            n_jobs=-1
        )
        rfecv.fit(X_scaled, y)

        selected = [f for f, s in zip(top_35, rfecv.support_) if s]

        if len(selected) < self.max_features:
            remaining = [f for f in top_35 if f not in selected]
            selected.extend(remaining[:self.max_features - len(selected)])
        elif len(selected) > self.max_features:
            selected = selected[:self.max_features]

        self.selected_features = selected
        print(f"Final selected: {len(selected)} features")
        return selected

    def transform(self, df):
        return df[self.selected_features]


# ==================== 7. 主流程（单进程+进度条） ====================
def main():
    print("Loading data...")
    df = load_data()

    all_dates = sorted(df['date'].dt.date.unique())
    all_dates = [d for d in all_dates if d > all_dates[0]]  # 跳过第一天
    print(f"Processing {len(all_dates)} days (skipped first day)")

    # 预计算
    print("\nInitializing precomputed stats...")
    precomputed = PrecomputedStats(df)

    # 单进程循环（带进度条，避免内存爆炸）
    print(f"\nGenerating samples and features...")
    all_samples = []

    for date in tqdm(all_dates, desc="Processing days"):
        try:
            # 生成当天样本
            samples = generate_daily_samples(df, date, neg_ratio=3)
            if len(samples) == 0:
                continue

            # 生成特征
            fe = FastFeatureEngineer(precomputed, date)
            featured_samples = fe.generate_features(samples)
            all_samples.append(featured_samples)

        except Exception as e:
            print(f"\nError on {date}: {e}")
            continue

    if len(all_samples) == 0:
        print("No samples generated!")
        return None, None, None, None

    full_data = pd.concat(all_samples, ignore_index=True)
    print(f"\nTotal samples: {len(full_data)}")
    print(f"Positive rate: {full_data['label'].mean():.4f}")

    # 划分
    print("\nSplitting (daily 8:1:1)...")
    train, val, test = split_daily_8_1_1(full_data)

    print(f"Train: {len(train)} ({len(train) / len(full_data):.1%})")
    print(f"Val: {len(val)} ({len(val) / len(full_data):.1%})")
    print(f"Test: {len(test)} ({len(test) / len(full_data):.1%})")

    # 特征筛选
    print("\nFeature selection...")
    selector = FeatureSelector(max_features=30)
    selected_feats = selector.fit(train)

    # 准备数据
    X_train = selector.transform(train).fillna(0)
    y_train = train['label']
    X_val = selector.transform(val).fillna(0)
    y_val = val['label']
    X_test = selector.transform(test).fillna(0)
    y_test = test['label']

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 训练
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1, penalty='l2')
    lr.fit(X_train_s, y_train)

    # 评估
    for name, X, y in [('Validation', X_val_s, y_val), ('Test', X_test_s, y_test)]:
        prob = lr.predict_proba(X)[:, 1]
        pred = lr.predict(X)
        auc = roc_auc_score(y, prob)
        print(f"\n{name} AUC: {auc:.4f}")
        print(classification_report(y, pred, target_names=['No Buy', 'Buy'], digits=4))

    # 特征重要性
    coef_df = pd.DataFrame({
        'feature': selected_feats,
        'coef': lr.coef_[0],
        'abs_coef': np.abs(lr.coef_[0])
    }).sort_values('abs_coef', ascending=False)

    print("\n" + "=" * 60)
    print("Final Selected Features:")
    print(coef_df.to_string(index=False))
    print("=" * 60)

    return lr, selector, scaler, coef_df


if __name__ == "__main__":
    model, selector, scaler, importance = main()

#结果：
# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\feature_engineering_v3.py
# Loading data...
# Processing 30 days (skipped first day)
#
# Initializing precomputed stats...
# Precomputing daily statistics...
#   Computing item daily stats...
#   Computing user daily stats...
#   Computing category daily stats...
#   Computing UI daily stats...
#   Computing UC daily stats...
# Precomputation completed!
#
# Generating samples and features...
# Processing days: 100%|██████████| 30/30 [15:10<00:00, 30.34s/it]
#
# Total samples: 411232
# Positive rate: 0.2500
#
# Splitting (daily 8:1:1)...
# Train: 328974 (80.0%)
# Val: 41122 (10.0%)
# Test: 41136 (10.0%)
#
# Feature selection...
# After variance filter: 52
# Calculating IV...
# After IV filter: 30
# XGBoost selection...
# RFECV...
# Final selected: 30 features
#
# Training Logistic Regression...
#
# Validation AUC: 0.7419
#               precision    recall  f1-score   support
#
#       No Buy     0.8661    0.6773    0.7601     30971
#          Buy     0.4087    0.6805    0.5107     10151
#
#     accuracy                         0.6781     41122
#    macro avg     0.6374    0.6789    0.6354     41122
# weighted avg     0.7532    0.6781    0.6986     41122
#
#
# Test AUC: 0.7361
#               precision    recall  f1-score   support
#
#       No Buy     0.8674    0.6756    0.7596     31258
#          Buy     0.3960    0.6731    0.4987      9878
#
#     accuracy                         0.6750     41136
#    macro avg     0.6317    0.6744    0.6291     41136
# weighted avg     0.7542    0.6750    0.6969     41136
#
#
# ============================================================
# Final Selected Features:
#              feature      coef  abs_coef
#       i_browse_count -4.210258  4.210258
#      i_act_total_21d  4.152146  4.152146
#   c_cart_to_buy_rate  0.759267  0.759267
#         c_cart_count  0.465709  0.465709
#     u_purchase_count  0.440694  0.440694
#       c_act_total_7d -0.372183  0.372183
#     uc_total_actions -0.365422  0.365422
#        uc_cart_count  0.343118  0.343118
#       c_browse_count -0.342127  0.342127
#    c_popularity_rank  0.321754  0.321754
#      c_act_total_21d -0.313878  0.313878
#          u_total_all  0.258351  0.258351
#  u_total_actions_21d -0.237446  0.237446
#      u_total_actions -0.237446  0.237446
#       i_act_total_1d  0.220807  0.220807
#      c_act_total_15d  0.211710  0.211710
# i_browse_to_buy_rate  0.148881  0.148881
#    i_fav_to_buy_rate  0.146019  0.146019
#      uc_browse_count  0.141319  0.141319
#     c_purchase_count  0.124448  0.124448
#  u_total_actions_15d -0.121451  0.121451
#           uc_recency -0.111212  0.111212
#   u_total_actions_7d -0.109896  0.109896
#          c_fav_count  0.085343  0.085343
#   i_cart_to_buy_rate -0.081994  0.081994
#       i_act_total_7d -0.061190  0.061190
#   u_total_actions_1d  0.046379  0.046379
#      i_act_total_15d  0.036748  0.036748
#       i_act_total_3d  0.032731  0.032731
#   u_total_actions_3d -0.021399  0.021399
# ============================================================
#
# Process finished with exit code 0