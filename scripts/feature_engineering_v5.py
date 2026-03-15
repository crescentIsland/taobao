"""
淘宝用户行为预测 - 完整特征版（XGBoost筛选 + 学习曲线 + 特征输出）
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
from sklearn.model_selection import learning_curve
import xgboost as xgb
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 路径配置 ====================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
DATA_PATH = os.path.join(project_root, 'data', 'processed', 'user_action_processed.parquet')

# 新增：特征保存路径和图表保存路径
FEATURE_SAVE_DIR = os.path.join(project_root, 'data', 'features_for_lr1')
CHART_SAVE_DIR = os.path.join(project_root, 'charts')
os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)
os.makedirs(CHART_SAVE_DIR, exist_ok=True)


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


# ==================== 6. 特征筛选（增加详细输出） ====================
class FeatureSelector:
    def __init__(self, max_features=40):
        self.max_features = max_features
        self.selected_features = None
        self.scaler = StandardScaler()
        self.feature_groups = {}  # 存储特征分组信息

    def fit(self, train_df, target_col='label'):
        exclude_cols = ['user_id', 'item_id', 'item_category', 'label', 'predict_date']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols
                        and pd.api.types.is_numeric_dtype(train_df[c])]

        X = train_df[feature_cols].fillna(0)
        y = train_df[target_col]

        print(f"\n{'=' * 80}")
        print(f"原始特征总数: {len(feature_cols)}")
        print(f"{'=' * 80}")

        # 方差过滤
        variances = X.var()
        valid_feats = variances[variances > 0.0001].index.tolist()
        if len(valid_feats) < 20:
            valid_feats = feature_cols
        X = X[valid_feats]
        print(f"方差过滤后: {len(valid_feats)} 个特征")

        # IV过滤
        print(f"\n计算 IV (Information Value)...")
        iv_scores = {}
        for col in tqdm(valid_feats, desc="IV计算"):
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
        print(f"IV过滤后: {len(iv_selected)} 个特征")

        # XGBoost初筛
        print(f"\nXGBoost 初筛 (Top 45)...")
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

        # RFECV精选
        print(f"\nRFECV 精选至 {self.max_features} 个特征...")
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

        # 对特征进行分组（便于理解）
        self._categorize_features(selected)

        # 打印详细特征列表
        self._print_selected_features(selected, importance)

        return selected

    def _categorize_features(self, features):
        """将特征按业务含义分组"""
        groups = {
            'Item Features (商品特征)': [f for f in features if f.startswith('i_')],
            'User Features (用户特征)': [f for f in features if
                                         f.startswith('u_') and not f.startswith('ui_') and not f.startswith('uc_')],
            'User-Item Interaction (用户-商品交互)': [f for f in features if f.startswith('ui_')],
            'Category Features (类目特征)': [f for f in features if f.startswith('c_')],
            'User-Category Interaction (用户-类目交互)': [f for f in features if f.startswith('uc_')],
            'Time Features (时间特征)': [f for f in features if 'hour' in f or 'time' in f or 'date' in f],
            'Conversion/Ratio (转化率)': [f for f in features if 'rate' in f or 'score' in f or 'concentration' in f],
            'Other (其他)': []
        }

        # 将未分类的特征放入Other
        categorized = set()
        for group in groups.values():
            categorized.update(group)

        for f in features:
            if f not in categorized:
                groups['Other (其他)'].append(f)

        self.feature_groups = {k: v for k, v in groups.items() if v}

    def _print_selected_features(self, selected, importance_df):
        """打印详细的特征列表"""
        print(f"\n{'=' * 80}")
        print(f"最终筛选出的 {len(selected)} 个特征 (按重要性排序)")
        print(f"{'=' * 80}")

        # 创建特征到重要性的映射
        imp_map = dict(zip(importance_df['feature'], importance_df['importance']))

        # 按分组打印
        for group_name, feats in self.feature_groups.items():
            if feats:
                print(f"\n【{group_name}】({len(feats)}个)")
                print("-" * 60)
                # 按重要性排序
                sorted_feats = sorted(feats, key=lambda x: imp_map.get(x, 0), reverse=True)
                for i, feat in enumerate(sorted_feats, 1):
                    imp = imp_map.get(feat, 0)
                    print(f"  {i:2d}. {feat:40s} (重要性: {imp:.4f})")

        print(f"\n{'=' * 80}")
        print("特征列表已保存至: feature_list.txt")
        print(f"{'=' * 80}\n")

    def transform(self, df):
        return df[self.selected_features]


# ==================== 7. 学习曲线绘制 ====================
def plot_learning_curves(estimator, X_train, y_train, X_val, y_val, save_dir):
    """
    绘制两种学习曲线：
    1. 随训练迭代次数的学习曲线（XGBoost evals_result）
    2. 随训练样本大小的学习曲线（sklearn learning_curve）
    """
    print(f"\n绘制学习曲线...")

    # 图1: 训练过程中的AUC变化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('XGBoost Learning Curves', fontsize=16, fontweight='bold')

    # 获取evals_result
    evals_result = estimator.evals_result()
    if evals_result:
        epochs = len(evals_result['validation_0']['auc'])
        x_axis = range(0, epochs)

        ax1 = axes[0, 0]
        ax1.plot(x_axis, evals_result['validation_0']['auc'], label='Validation AUC', linewidth=2)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('AUC')
        ax1.set_title('AUC during Training (Validation Set)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 图2: 特征重要性Top 20
    ax2 = axes[0, 1]
    importance = pd.DataFrame({
        'feature': estimator.feature_names_in_,
        'importance': estimator.feature_importances_
    }).sort_values('importance', ascending=True).tail(20)

    ax2.barh(importance['feature'], importance['importance'])
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 20 Feature Importances (XGBoost)')
    ax2.grid(True, alpha=0.3, axis='x')

    # 图3: 随样本大小的学习曲线（使用5个关键点）
    ax3 = axes[1, 0]
    train_sizes = np.linspace(0.1, 1.0, 5)

    train_scores = []
    val_scores = []

    for frac in train_sizes:
        n_samples = int(len(X_train) * frac)
        if n_samples < 100:
            continue

        X_subset = X_train.iloc[:n_samples]
        y_subset = y_train.iloc[:n_samples]

        # 快速训练一个简单模型评估
        temp_model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        temp_model.fit(X_subset, y_subset, verbose=False)

        train_pred = temp_model.predict_proba(X_subset)[:, 1]
        val_pred = temp_model.predict_proba(X_val)[:, 1]

        train_scores.append(roc_auc_score(y_subset, train_pred))
        val_scores.append(roc_auc_score(y_val, val_pred))

    ax3.plot(train_sizes[:len(train_scores)], train_scores, 'o-', label='Training AUC', linewidth=2)
    ax3.plot(train_sizes[:len(train_scores)], val_scores, 'o-', label='Validation AUC', linewidth=2)
    ax3.set_xlabel('Training Set Size (fraction)')
    ax3.set_ylabel('AUC Score')
    ax3.set_title('Learning Curve (Sample Size vs Performance)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: 过拟合分析（训练集vs验证集性能差距）
    ax4 = axes[1, 1]
    gaps = [t - v for t, v in zip(train_scores, val_scores)]
    ax4.bar(range(len(gaps)), gaps, color=['green' if g < 0.05 else 'orange' if g < 0.1 else 'red' for g in gaps])
    ax4.set_xlabel('Training Size Index')
    ax4.set_ylabel('AUC Gap (Train - Val)')
    ax4.set_title('Overfitting Analysis (Gap < 0.05: Good)')
    ax4.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Warning threshold')
    ax4.axhline(y=0.1, color='darkred', linestyle='--', alpha=0.5, label='Overfitting threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ 学习曲线已保存: {chart_path}")
    plt.close()

    # 额外：生成详细的CSV报告
    report_data = {
        'training_fraction': train_sizes[:len(train_scores)],
        'train_auc': train_scores,
        'val_auc': val_scores,
        'gap': gaps,
        'status': ['Good' if g < 0.05 else 'Warning' if g < 0.1 else 'Overfitting' for g in gaps]
    }
    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(save_dir, 'learning_curve_data.csv')
    report_df.to_csv(report_path, index=False)
    print(f"✓ 学习曲线数据已保存: {report_path}")


# ==================== 8. 特征保存 ====================
def save_features_for_lr(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, save_dir, selector):
    """
    保存筛选后的特征集用于后续逻辑回归，包含分组信息
    """
    print(f"\n保存特征用于逻辑回归: {save_dir}")

    train_df = X_train.copy()
    train_df['label'] = y_train.values
    val_df = X_val.copy()
    val_df['label'] = y_val.values
    test_df = X_test.copy()
    test_df['label'] = y_test.values

    train_path = os.path.join(save_dir, 'train_features.parquet')
    val_path = os.path.join(save_dir, 'val_features.parquet')
    test_path = os.path.join(save_dir, 'test_features.parquet')

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # 保存详细特征信息（包含分组）
    feature_info = {
        'selected_features': feature_names,
        'feature_groups': selector.feature_groups,
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

    # 保存格式化的特征列表
    txt_path = os.path.join(save_dir, 'feature_list.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=" * 70 + "\n")
        f.write(f"筛选出的 {len(feature_names)} 个特征 (淘宝用户购买预测)\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 70 + "\n\n")

        for group_name, feats in selector.feature_groups.items():
            if feats:
                f.write(f"【{group_name}】({len(feats)}个)\n")
                f.write("-" * 50 + "\n")
                for i, feat in enumerate(feats, 1):
                    f.write(f"{i:2d}. {feat}\n")
                f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("数据文件信息:\n")
        f.write(f"- 训练集: {len(train_df)} 样本\n")
        f.write(f"- 验证集: {len(val_df)} 样本\n")
        f.write(f"- 测试集: {len(test_df)} 样本\n")
        f.write(f"- 正样本比例: {y_train.mean():.2%}\n")

    print(f"✓ 训练集: {train_path} ({len(train_df)} samples)")
    print(f"✓ 验证集: {val_path} ({len(val_df)} samples)")
    print(f"✓ 测试集: {test_path} ({len(test_df)} samples)")
    print(f"✓ 特征详情: {txt_path}")

    return train_path, val_path, test_path


# ==================== 9. 主流程 ====================
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
        return None, None, None

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

    # 特征筛选（40个，带详细输出）
    print("\n" + "=" * 80)
    print("开始特征筛选...")
    print("=" * 80)

    selector = FeatureSelector(max_features=40)
    selected_feats = selector.fit(train)

    # 准备数据
    X_train = selector.transform(train).fillna(0)
    y_train = train['label']
    X_val = selector.transform(val).fillna(0)
    y_val = val['label']
    X_test = selector.transform(test).fillna(0)
    y_test = test['label']

    # 保存特征
    save_features_for_lr(
        X_train, y_train, X_val, y_val, X_test, y_test,
        selected_feats, FEATURE_SAVE_DIR, selector
    )

    # XGBoost训练（启用evals_result以绘制学习曲线）
    print("\nTraining XGBoost with learning curve tracking...")
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

    # 评估
    print("\n" + "=" * 60)
    print("XGBoost 评估结果:")
    print("=" * 60)
    for name, X, y in [('验证集', X_val, y_val), ('测试集', X_test, y_test)]:
        prob = xgb_model.predict_proba(X)[:, 1]
        pred = xgb_model.predict(X)
        auc = roc_auc_score(y, prob)
        print(f"\n{name} AUC: {auc:.4f}")
        print(classification_report(y, pred, target_names=['未购买', '购买'], digits=4))

    # 绘制学习曲线
    plot_learning_curves(xgb_model, X_train, y_train, X_val, y_val, CHART_SAVE_DIR)

    # 保存最终重要性
    importance = pd.DataFrame({
        'feature': selected_feats,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance.to_csv(os.path.join(FEATURE_SAVE_DIR, 'xgboost_feature_importance.csv'), index=False)

    print("\n" + "=" * 60)
    print("Top 40 重要特征:")
    print(importance.head(40).to_string(index=False))
    print("=" * 60)

    return xgb_model, selector, importance


if __name__ == "__main__":
    model, selector, importance = main()