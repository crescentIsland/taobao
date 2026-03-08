"""
淘宝用户行为预测 - 完整特征版（XGBoost筛选 + 保存特征用于LR）
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import RFECV
import xgboost as xgb
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. 路径配置 ====================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
DATA_PATH = os.path.join(project_root, 'data', 'processed', 'user_action_processed.parquet')

# 新增：特征保存路径
FEATURE_SAVE_DIR = os.path.join(project_root, 'data', 'features_for_lr')
os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)


# [保持之前的 PrecomputedStats 和 CompleteFeatureEngineer 类不变...]
# [保持 generate_daily_samples 和 split_daily_8_1_1 函数不变...]
# [保持 FeatureSelector 类不变...]

# ==================== 2. 数据加载 ====================
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['datetime'].dt.hour
    return df


# ==================== 3. 预计算引擎 ====================
class PrecomputedStats:
    def __init__(self, df):
        print("Precomputing comprehensive statistics...")
        self.df = df.copy()
        self.all_dates = sorted(df['date'].dt.date.unique())

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

        print("  Computing global stats...")
        self.global_stats = self._compute_global_stats()

        print("Precomputation completed!")

    def _compute_item_daily(self):
        stats = self.df.groupby(['item_id', 'date']).agg({
            'behavior_type': [
                ('browse', lambda x: (x == '1').sum()),
                ('fav', lambda x: (x == '2').sum()),
                ('cart', lambda x: (x == '3').sum()),
                ('purchase', lambda x: (x == '4').sum())
            ],
            'hour': [('hot_hour', lambda x: x.value_counts().index[0] if len(x) > 0 else 12)],
            'user_id': [('unique_users', 'nunique')]
        }).reset_index()
        stats.columns = ['item_id', 'date', 'browse', 'fav', 'cart', 'purchase', 'hot_hour', 'unique_users']
        return stats

    def _compute_user_daily(self):
        stats = self.df.groupby(['user_id', 'date']).agg({
            'behavior_type': [
                ('total', 'count'),
                ('purchase', lambda x: (x == '4').sum())
            ],
            'hour': [('hour_mode', lambda x: x.value_counts().index[0] if len(x) > 0 else 12)]
        }).reset_index()
        stats.columns = ['user_id', 'date', 'total', 'purchase', 'hour_mode']

        purchase_times = self.df[self.df['behavior_type'] == '4'].groupby(['user_id', 'date']).size().reset_index(
            name='p_count')
        purchase_times = purchase_times.sort_values(['user_id', 'date'])
        purchase_times['next_purchase'] = purchase_times.groupby('user_id')['date'].shift(-1)
        purchase_times['interval'] = (purchase_times['next_purchase'] - purchase_times['date']).dt.days
        self.user_purchase_interval = purchase_times.groupby('user_id')['interval'].mean().reset_index(
            name='avg_interval')

        return stats

    def _compute_category_daily(self):
        stats = self.df.groupby(['item_category', 'date']).agg({
            'behavior_type': [
                ('browse', lambda x: (x == '1').sum()),
                ('fav', lambda x: (x == '2').sum()),
                ('cart', lambda x: (x == '3').sum()),
                ('purchase', lambda x: (x == '4').sum())
            ]
        }).reset_index()
        stats.columns = ['item_category', 'date', 'browse', 'fav', 'cart', 'purchase']
        return stats

    def _compute_ui_daily(self):
        stats = self.df.groupby(['user_id', 'item_id', 'date']).agg({
            'behavior_type': [
                ('total', 'count'),
                ('view', lambda x: (x == '1').sum()),
                ('fav', lambda x: (x == '2').sum()),
                ('cart', lambda x: (x == '3').sum()),
                ('purchase', lambda x: (x == '4').sum())
            ],
            'hour': [('last_hour', 'last')]
        }).reset_index()
        stats.columns = ['user_id', 'item_id', 'date', 'total', 'view', 'fav', 'cart', 'purchase', 'last_hour']
        return stats

    def _compute_uc_daily(self):
        stats = self.df.groupby(['user_id', 'item_category', 'date']).agg({
            'behavior_type': [
                ('total', 'count'),
                ('purchase', lambda x: (x == '4').sum()),
                ('fav', lambda x: (x == '2').sum()),
                ('cart', lambda x: (x == '3').sum()),
                ('browse', lambda x: (x == '1').sum())
            ]
        }).reset_index()
        stats.columns = ['user_id', 'item_category', 'date', 'total', 'purchase', 'fav', 'cart', 'browse']
        return stats

    def _compute_global_stats(self):
        cat_total = self.cat_daily.groupby('item_category')[['browse', 'fav', 'cart', 'purchase']].sum().sum(axis=1)
        top_cats = cat_total.nlargest(10).index.tolist()
        return {'top_categories': set(top_cats)}


# ==================== 4. 完整特征工程 ====================
class CompleteFeatureEngineer:
    WINDOWS = [1, 3, 7, 14, 21]

    def __init__(self, precomputed, predict_date):
        self.pc = precomputed
        self.date = pd.Timestamp(predict_date)
        self.hist_dates = [self.date - timedelta(days=i) for i in range(1, 22)]

    def generate_features(self, samples):
        result = samples.copy()
        result = self._add_item_features(result)
        result = self._add_user_features(result)
        result = self._add_ui_features(result)
        result = self._add_category_features(result)
        result = self._add_uc_features(result)
        result = self._add_time_features(result)
        result = self._add_collaborative_features(result)
        result = self._add_conversion_features(result)
        result = self._add_rare_features(result)
        result = self._add_item_category_features(result)
        return result

    def _add_item_features(self, df):
        hist = self.pc.item_daily[self.pc.item_daily['date'].isin(self.hist_dates)]

        item_hist = hist.groupby('item_id').agg({
            'browse': 'sum', 'fav': 'sum', 'cart': 'sum', 'purchase': 'sum',
            'hot_hour': 'last', 'unique_users': 'sum', 'date': 'max'
        }).reset_index()
        item_hist.columns = ['item_id', 'i_browse_count', 'i_fav_count', 'i_cart_count',
                             'i_purchase_count', 'i_hot_hour', 'i_unique_users', 'last_date']

        item_hist['i_browse_to_buy_rate'] = (item_hist['i_purchase_count'] + 1) / (item_hist['i_browse_count'] + 10)
        item_hist['i_fav_to_buy_rate'] = (item_hist['i_purchase_count'] + 1) / (item_hist['i_fav_count'] + 10)
        item_hist['i_cart_to_buy_rate'] = (item_hist['i_purchase_count'] + 1) / (item_hist['i_cart_count'] + 10)
        item_hist['i_days_since_last_action'] = (self.date - item_hist['last_date']).dt.days

        for window in self.WINDOWS:
            window_dates = self.hist_dates[:window]
            window_data = hist[hist['date'].isin(window_dates)]
            if len(window_data) > 0:
                window_agg = window_data.groupby('item_id')[['browse', 'fav', 'cart', 'purchase']].sum()
                window_agg['total'] = window_agg.sum(axis=1)
                window_agg = window_agg.reset_index()[['item_id', 'total']]
                window_agg.columns = ['item_id', f'i_act_total_{window}d']
                item_hist = item_hist.merge(window_agg, on='item_id', how='left')
            else:
                item_hist[f'i_act_total_{window}d'] = 0

        item_hist['i_is_popular'] = (item_hist['i_unique_users'] > item_hist['i_unique_users'].quantile(0.9)).astype(
            int)

        merge_cols = [c for c in item_hist.columns if c != 'last_date']
        result = df.merge(item_hist[merge_cols], on='item_id', how='left')
        return result.fillna(0)

    def _add_user_features(self, df):
        hist = self.pc.user_daily[self.pc.user_daily['date'].isin(self.hist_dates)]

        user_hist = hist.groupby('user_id').agg({
            'total': 'sum', 'purchase': 'sum', 'hour_mode': 'last'
        }).reset_index()
        user_hist.columns = ['user_id', 'u_total_actions', 'u_purchase_count', 'u_preferred_hour']
        user_hist['u_purchase_rate'] = user_hist['u_purchase_count'] / (user_hist['u_total_actions'] + 1)

        for window in self.WINDOWS:
            window_dates = self.hist_dates[:window]
            window_data = hist[hist['date'].isin(window_dates)]
            if len(window_data) > 0:
                window_agg = window_data.groupby('user_id')['total'].sum().reset_index()
                window_agg.columns = ['user_id', f'u_total_actions_{window}d']
                user_hist = user_hist.merge(window_agg, on='user_id', how='left')
            else:
                user_hist[f'u_total_actions_{window}d'] = 0

        user_hist = user_hist.merge(self.pc.user_purchase_interval, on='user_id', how='left')
        user_hist['avg_interval'] = user_hist['avg_interval'].fillna(30)

        result = df.merge(user_hist, on='user_id', how='left')
        return result.fillna(0)

    def _add_ui_features(self, df):
        hist = self.pc.ui_daily[self.pc.ui_daily['date'].isin(self.hist_dates)]

        if len(hist) == 0:
            for col in ['ui_total_actions', 'ui_view_count', 'ui_fav_count', 'ui_cart_count',
                        'ui_has_carted', 'ui_has_faved', 'ui_days_since_last_action',
                        'ui_sequence_length', 'ui_has_fav_cart_pattern', 'ui_is_today',
                        'ui_funnel_score', 'ui_attention_intensity', 'ui_last_action_hour',
                        'ui_recent_action_concentration', 'ui_time_from_first_view', 'ui_view_frequency']:
                df[col] = 0
            return df

        ui_hist = hist.groupby(['user_id', 'item_id']).agg({
            'total': 'sum', 'view': 'sum', 'fav': 'sum', 'cart': 'sum', 'purchase': 'sum',
            'date': ['min', 'max', 'nunique'], 'last_hour': 'last'
        }).reset_index()
        ui_hist.columns = ['user_id', 'item_id', 'ui_total_actions', 'ui_view_count',
                           'ui_fav_count', 'ui_cart_count', 'ui_purchase_count_hist',
                           'first_date', 'last_date', 'ui_view_days', 'ui_last_action_hour']

        ui_hist['ui_has_carted'] = (ui_hist['ui_cart_count'] > 0).astype(int)
        ui_hist['ui_has_faved'] = (ui_hist['ui_fav_count'] > 0).astype(int)
        ui_hist['ui_days_since_last_action'] = (self.date - ui_hist['last_date']).dt.days
        ui_hist['ui_is_today'] = (ui_hist['ui_days_since_last_action'] == 0).astype(int)
        ui_hist['ui_sequence_length'] = ui_hist[['ui_view_count', 'ui_fav_count', 'ui_cart_count']].sum(axis=1)
        ui_hist['ui_has_fav_cart_pattern'] = ((ui_hist['ui_fav_count'] > 0) & (ui_hist['ui_cart_count'] > 0)).astype(
            int)
        ui_hist['ui_funnel_score'] = (ui_hist['ui_view_count'] * 0.1 +
                                      ui_hist['ui_fav_count'] * 0.2 +
                                      ui_hist['ui_cart_count'] * 0.3)

        window_3d = self.hist_dates[:3]
        recent_hist = hist[hist['date'].isin(window_3d)]
        if len(recent_hist) > 0:
            recent_ui = recent_hist.groupby(['user_id', 'item_id'])['total'].sum().reset_index(name='ui_recent_3d')
            ui_hist = ui_hist.merge(recent_ui, on=['user_id', 'item_id'], how='left')
            ui_hist['ui_recent_3d'] = ui_hist['ui_recent_3d'].fillna(0)
        else:
            ui_hist['ui_recent_3d'] = 0

        ui_hist['ui_time_from_first_view'] = (self.date - ui_hist['first_date']).dt.total_seconds() / 3600
        ui_hist['ui_view_frequency'] = ui_hist['ui_view_count'] / ui_hist['ui_view_days'].clip(lower=1)

        result = df.merge(ui_hist.drop(['first_date', 'last_date'], axis=1),
                          on=['user_id', 'item_id'], how='left')
        return result.fillna(0)

    def _add_category_features(self, df):
        hist = self.pc.cat_daily[self.pc.cat_daily['date'].isin(self.hist_dates)]

        cat_hist = hist.groupby('item_category').agg({
            'browse': 'sum', 'fav': 'sum', 'cart': 'sum', 'purchase': 'sum'
        }).reset_index()
        cat_hist.columns = ['item_category', 'c_browse_count', 'c_fav_count',
                            'c_cart_count', 'c_purchase_count']

        cat_hist['c_browse_to_buy_rate'] = (cat_hist['c_purchase_count'] + 1) / (cat_hist['c_browse_count'] + 10)
        cat_hist['c_fav_to_buy_rate'] = (cat_hist['c_purchase_count'] + 1) / (cat_hist['c_fav_count'] + 10)
        cat_hist['c_cart_to_buy_rate'] = (cat_hist['c_purchase_count'] + 1) / (cat_hist['c_cart_count'] + 10)

        for window in self.WINDOWS:
            window_dates = self.hist_dates[:window]
            window_data = hist[hist['date'].isin(window_dates)]
            if len(window_data) > 0:
                window_agg = window_data.groupby('item_category')[['browse', 'fav', 'cart', 'purchase']].sum().sum(
                    axis=1)
                window_agg = window_agg.reset_index()
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
            'total': 'sum', 'purchase': 'sum', 'fav': 'sum', 'cart': 'sum', 'browse': 'sum',
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

        result = df.merge(uc_hist.drop('last_date', axis=1), on=['user_id', 'item_category'], how='left')
        return result.fillna(0)

    def _add_time_features(self, df):
        df['ui_hour_diff'] = abs(df['u_preferred_hour'] - df['i_hot_hour'])
        df['ui_hour_match'] = (df['ui_hour_diff'] <= 2).astype(int)
        return df

    def _add_collaborative_features(self, df):
        df['ui_popularity_preference_score'] = df['i_act_total_7d'] * df['uc_preference_score']
        return df

    def _add_conversion_features(self, df):
        df['ui_recent_action_concentration'] = df['ui_recent_3d'] / (df['u_total_actions_3d'] + 1)
        df['uc_category_share'] = df['uc_total_actions'] / (df['u_total_actions'] + 1)
        df['ui_attention_intensity'] = df['ui_total_actions'] / (
                    df['u_total_actions'] / (df.groupby('user_id')['item_id'].transform('count') + 1))
        return df

    def _add_rare_features(self, df):
        df['ui_is_only_viewer'] = ((df['i_unique_users'] == 1) & (df['ui_view_count'] > 0)).astype(int)
        df['ui_is_impulse_buy_candidate'] = ((df['ui_view_count'] <= 2) & (df['ui_has_carted'] == 1)).astype(int)
        return df

    def _add_item_category_features(self, df):
        df['i_in_top_category'] = df['item_category'].apply(
            lambda x: 1 if x in self.pc.global_stats['top_categories'] else 0
        )
        return df


# ==================== 5. 样本生成与划分 ====================
def generate_daily_samples(df, date, neg_ratio=3):
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
    def __init__(self, max_features=40):
        self.max_features = max_features
        self.selected_features = None
        self.scaler = StandardScaler()

    def fit(self, train_df, target_col='label'):
        exclude_cols = ['user_id', 'item_id', 'item_category', 'label', 'predict_date']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols
                        and pd.api.types.is_numeric_dtype(train_df[c])]

        X = train_df[feature_cols].fillna(0)
        y = train_df[target_col]

        print(f"Original features: {len(feature_cols)}")

        variances = X.var()
        valid_feats = variances[variances > 0.0001].index.tolist()
        if len(valid_feats) < 20:
            valid_feats = feature_cols
        X = X[valid_feats]
        print(f"After variance filter: {len(valid_feats)}")

        print("Calculating IV...")
        iv_scores = {}
        for col in valid_feats:
            try:
                x_clean = X[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                if x_clean.nunique() < 2:
                    continue
                bins = pd.qcut(x_clean, q=10, duplicates='drop')
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
        iv_selected = iv_df[iv_df['iv'] > 0.01]['feature'].tolist()
        if len(iv_selected) > 50:
            iv_selected = iv_df.head(50)['feature'].tolist()
        if len(iv_selected) < 10:
            iv_selected = valid_feats[:min(30, len(valid_feats))]
        print(f"After IV filter: {len(iv_selected)}")

        print("XGBoost selection...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        xgb_model.fit(X[iv_selected], y)

        importance = pd.DataFrame({
            'feature': iv_selected,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        top_45 = importance.head(45)['feature'].tolist()

        print("RFECV...")
        X_scaled = self.scaler.fit_transform(X[top_45])

        xgb_for_rfecv = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )

        rfecv = RFECV(
            estimator=xgb_for_rfecv,
            step=1,
            cv=3,
            scoring='roc_auc',
            min_features_to_select=self.max_features,
            n_jobs=-1
        )
        rfecv.fit(X_scaled, y)

        selected = [f for f, s in zip(top_45, rfecv.support_) if s]

        if len(selected) < self.max_features:
            remaining = [f for f in importance['feature'] if f not in selected]
            selected.extend(remaining[:self.max_features - len(selected)])
        elif len(selected) > self.max_features:
            selected = selected[:self.max_features]

        self.selected_features = selected
        print(f"Final selected: {len(selected)} features")
        return selected

    def transform(self, df):
        return df[self.selected_features]


# ==================== 7. 主流程（增加保存功能） ====================
def save_features_for_lr(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, save_dir):
    """
    保存筛选后的特征集用于后续逻辑回归
    保存格式：Parquet（高效）+ JSON（特征名）
    """
    print(f"\nSaving features for Logistic Regression to: {save_dir}")

    # 组合X和y保存（保留ID列用于后续分析）
    train_df = X_train.copy()
    train_df['label'] = y_train.values

    val_df = X_val.copy()
    val_df['label'] = y_val.values

    test_df = X_test.copy()
    test_df['label'] = y_test.values

    # 保存为Parquet（比CSV快10倍，体积小5倍）
    train_path = os.path.join(save_dir, 'train_features.parquet')
    val_path = os.path.join(save_dir, 'val_features.parquet')
    test_path = os.path.join(save_dir, 'test_features.parquet')

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # 保存特征名列表（JSON格式）
    feature_info = {
        'selected_features': feature_names,
        'n_features': len(feature_names),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'positive_rate_train': float(y_train.mean()),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    json_path = os.path.join(save_dir, 'feature_info.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)

    # 同时保存一份CSV格式的特征名方便查看
    txt_path = os.path.join(save_dir, 'feature_list.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Selected Features ({len(feature_names)} total):\n")
        f.write("=" * 60 + "\n")
        for i, feat in enumerate(feature_names, 1):
            f.write(f"{i:2d}. {feat}\n")

    print(f"✓ Train set: {train_path} ({len(train_df)} samples, {len(feature_names)} features)")
    print(f"✓ Val set:   {val_path} ({len(val_df)} samples)")
    print(f"✓ Test set:  {test_path} ({len(test_df)} samples)")
    print(f"✓ Feature list: {txt_path}")

    return train_path, val_path, test_path


def main():
    print("Loading data...")
    df = load_data()

    all_dates = sorted(df['date'].dt.date.unique())
    all_dates = [d for d in all_dates if d > all_dates[0]]
    print(f"Processing {len(all_dates)} days")

    print("\nInitializing precomputed stats...")
    precomputed = PrecomputedStats(df)

    print(f"\nGenerating samples and features...")
    all_samples = []

    for date in tqdm(all_dates, desc="Processing days"):
        try:
            samples = generate_daily_samples(df, date, neg_ratio=3)
            if len(samples) == 0:
                continue

            fe = CompleteFeatureEngineer(precomputed, date)
            featured_samples = fe.generate_features(samples)
            all_samples.append(featured_samples)

        except Exception as e:
            print(f"\nError on {date}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_samples) == 0:
        print("No samples generated!")
        return None, None, None, None

    full_data = pd.concat(all_samples, ignore_index=True)
    print(f"\nTotal samples: {len(full_data)}")
    numeric_cols = [c for c in full_data.columns if
                    c not in ['user_id', 'item_id', 'item_category', 'label', 'predict_date']]
    print(f"Features generated: {len(numeric_cols)}")
    print(f"Positive rate: {full_data['label'].mean():.4f}")

    print("\nSplitting (daily 8:1:1)...")
    train, val, test = split_daily_8_1_1(full_data)

    print(f"Train: {len(train)} ({len(train) / len(full_data):.1%})")
    print(f"Val: {len(val)} ({len(val) / len(full_data):.1%})")
    print(f"Test: {len(test)} ({len(test) / len(full_data):.1%})")

    # 特征筛选（40个）
    print("\nFeature selection...")
    selector = FeatureSelector(max_features=40)
    selected_feats = selector.fit(train)

    # 准备数据（未标准化，保留原始值给后续LR使用）
    X_train = selector.transform(train).fillna(0)
    y_train = train['label']
    X_val = selector.transform(val).fillna(0)
    y_val = val['label']
    X_test = selector.transform(test).fillna(0)
    y_test = test['label']

    # ==================== 新增：保存特征 ====================
    save_features_for_lr(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        selected_feats,
        FEATURE_SAVE_DIR
    )
    # ======================================================

    # XGBoost训练（使用同样的40个特征进行对比）
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=20
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 评估XGBoost
    print("\n" + "=" * 60)
    print("XGBoost Results:")
    for name, X, y in [('Validation', X_val, y_val), ('Test', X_test, y_test)]:
        prob = xgb_model.predict_proba(X)[:, 1]
        pred = xgb_model.predict(X)
        auc = roc_auc_score(y, prob)
        print(f"\n{name} AUC: {auc:.4f}")
        print(classification_report(y, pred, target_names=['No Buy', 'Buy'], digits=4))

    # 特征重要性
    importance = pd.DataFrame({
        'feature': selected_feats,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Important Features (XGBoost):")
    print(importance.head(20).to_string(index=False))
    print("=" * 60)

    # 保存XGBoost重要性供LR参考
    importance.to_csv(os.path.join(FEATURE_SAVE_DIR, 'xgboost_feature_importance.csv'), index=False)

    return xgb_model, selector, importance


if __name__ == "__main__":
    model, selector, importance = main()

# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\feature_engineering_v4.py
# Loading data...
# Processing 30 days
#
# Initializing precomputed stats...
# Precomputing comprehensive statistics...
#   Computing item daily stats...
#   Computing user daily stats...
#   Computing category daily stats...
#   Computing UI daily stats...
#   Computing UC daily stats...
#   Computing global stats...
# Precomputation completed!
#
# Generating samples and features...
# Processing days: 100%|██████████| 30/30 [17:32<00:00, 35.07s/it]
#
# Total samples: 411232
# Features generated: 72
# Positive rate: 0.2500
#
# Splitting (daily 8:1:1)...
# Train: 328974 (80.0%)
# Val: 41122 (10.0%)
# Test: 41136 (10.0%)
#
# Feature selection...
# Original features: 72
# After variance filter: 71
# Calculating IV...
# After IV filter: 50
# XGBoost selection...
# RFECV...
# Final selected: 40 features
#
# Saving features for Logistic Regression to: E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr
# ✓ Train set: E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr\train_features.parquet (328974 samples, 40 features)
# ✓ Val set:   E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr\val_features.parquet (41122 samples)
# ✓ Test set:  E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr\test_features.parquet (41136 samples)
# ✓ Feature list: E:\PycharmeProjects\taobao_user_behavior_analysis\data\features_for_lr\feature_list.txt
#
# Training XGBoost...
#
# ============================================================
# XGBoost Results:
#
# Validation AUC: 0.8214
#               precision    recall  f1-score   support
#
#       No Buy     0.9010    0.7217    0.8015     30971
#          Buy     0.4717    0.7582    0.5816     10151
#
#     accuracy                         0.7307     41122
#    macro avg     0.6864    0.7399    0.6915     41122
# weighted avg     0.7951    0.7307    0.7472     41122
#
#
# Test AUC: 0.8201
#               precision    recall  f1-score   support
#
#       No Buy     0.9032    0.7251    0.8044     31258
#          Buy     0.4643    0.7540    0.5747      9878
#
#     accuracy                         0.7321     41136
#    macro avg     0.6838    0.7396    0.6896     41136
# weighted avg     0.7978    0.7321    0.7493     41136
#
#
# Top 20 Important Features (XGBoost):
#                        feature  importance
#                ui_funnel_score    0.162812
#           c_browse_to_buy_rate    0.154067
#         ui_attention_intensity    0.117733
#                u_purchase_rate    0.057006
# ui_recent_action_concentration    0.055910
#              c_fav_to_buy_rate    0.045614
#                 c_browse_count    0.030460
#                   i_cart_count    0.029125
#                   avg_interval    0.026188
#             i_cart_to_buy_rate    0.023095
#             u_total_actions_7d    0.021048
#      ui_days_since_last_action    0.019423
#                c_act_total_21d    0.019080
#            u_total_actions_14d    0.017890
#                   ui_recent_3d    0.014348
#            u_total_actions_21d    0.013715
#            uc_preference_score    0.013037
#              i_fav_to_buy_rate    0.012343
#               u_purchase_count    0.012136
#             u_total_actions_3d    0.011074
# ============================================================
#
# Process finished with exit code 0
