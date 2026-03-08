"""
全量数据特征工程（修正版8:1:1划分）
时间: 2014-11-18 至 2014-12-18（共31天）
划分: 训练25天 : 验证3天 : 测试3天
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
from tqdm import tqdm
import gc
import joblib

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

DATA_PATH = os.path.join(project_root, 'data', 'processed', 'user_action_processed.parquet')
RESULTS_DIR = os.path.join(project_root, 'results', 'full_features_selected')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 修正后的8:1:1时间划分（25:3:3天）
DATE_RANGES = {
    'train': (datetime(2014, 11, 18), datetime(2014, 12, 12)),  # 25天 (~81%)
    'val': (datetime(2014, 12, 13), datetime(2014, 12, 15)),  # 3天 (~10%)
    'test': (datetime(2014, 12, 16), datetime(2014, 12, 18))  # 3天 (~10%)
}

BEST_WINDOWS = [1, 3, 7, 14, 21]


class OptimizedFeatureEngineer:
    def __init__(self, train_data, current_time):
        self.train_data = train_data.copy()
        self.current_time = current_time
        self.start_time = time.time()

        self.train_data['days_diff'] = (
                                               self.current_time - self.train_data['behavior_time']
                                       ).dt.total_seconds() / 86400

        self.train_data['hour'] = self.train_data['behavior_time'].dt.hour
        self.train_data['time_slot'] = self.train_data['hour'] // 4
        self.train_data['is_weekend'] = self.train_data['behavior_time'].dt.dayofweek.isin([5, 6]).astype(int)

        self.window_masks = {}
        for w in BEST_WINDOWS:
            self.window_masks[w] = self.train_data['days_diff'] <= w

    def log(self, msg):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {msg}")

    def generate_all_features(self, labels):
        if len(labels) == 0:
            raise ValueError("标签为空")

        self.log("开始特征工程...")

        item_features = self._gen_item_features()
        gc.collect()

        user_features = self._gen_user_features()
        gc.collect()

        ui_features = self._gen_ui_features()
        gc.collect()

        time_features = self._gen_time_features_4h()
        gc.collect()

        advanced_features = self._gen_advanced_features(labels)
        gc.collect()

        self.log("合并所有特征...")
        features_df = labels.merge(
            item_features, left_on='item_id', right_index=True, how='left'
        ).merge(
            user_features, left_on='user_id', right_index=True, how='left'
        ).merge(
            ui_features, left_on=['user_id', 'item_id'], right_index=True, how='left'
        ).merge(
            time_features, left_on=['user_id', 'item_id'], right_index=True, how='left'
        ).merge(
            advanced_features, left_on=['user_id', 'item_id'], right_index=True, how='left'
        )

        features_df = self._calc_derived_features(features_df)
        features_df = features_df.fillna(0)

        self.log(f"特征生成完成: {features_df.shape[1] - 3}个特征")
        return features_df

    def _gen_item_features(self):
        self.log("生成商品特征...")

        item_stats = self.train_data.groupby('item_id').agg({
            'behavior_type': ['count', 'sum'],
            'user_id': 'nunique',
            'is_weekend': 'mean'
        })
        item_stats.columns = ['i_act_total', 'i_purchase_count', 'i_unique_users', 'i_weekend_ratio']

        behavior_counts = self.train_data.pivot_table(
            index='item_id', columns='behavior_type', values='user_id',
            aggfunc='count', fill_value=0
        )
        for btype, col_name in [(1, 'i_browse_count'), (2, 'i_fav_count'),
                                (3, 'i_cart_count'), (4, 'i_purchase_count')]:
            item_stats[col_name] = behavior_counts.get(btype, 0)

        for w in BEST_WINDOWS:
            mask = self.window_masks[w]
            window_data = self.train_data[mask]
            if len(window_data) > 0:
                item_stats[f'i_act_total_{w}d'] = window_data.groupby('item_id').size().reindex(
                    item_stats.index).fillna(0)
                for btype, prefix in [(1, 'browse'), (2, 'fav'), (3, 'cart'), (4, 'purchase')]:
                    cnt = window_data[window_data['behavior_type'] == btype].groupby('item_id').size()
                    item_stats[f'i_{prefix}_count_{w}d'] = cnt.reindex(item_stats.index).fillna(0)
            else:
                item_stats[f'i_act_total_{w}d'] = 0
                for prefix in ['browse', 'fav', 'cart', 'purchase']:
                    item_stats[f'i_{prefix}_count_{w}d'] = 0

        last_act = self.train_data.groupby('item_id')['behavior_time'].max()
        item_stats['i_days_since_last_action'] = (self.current_time - last_act).dt.total_seconds() / 86400

        # 趋势特征（近7天 vs 前14天）
        recent = self.train_data[self.train_data['days_diff'] <= 7].groupby('item_id').size()
        prev = self.train_data[(self.train_data['days_diff'] > 7) &
                               (self.train_data['days_diff'] <= 21)].groupby('item_id').size()
        item_stats['i_action_trend'] = recent.reindex(item_stats.index).fillna(0) / (
                prev.reindex(item_stats.index).fillna(0) + 1)

        return item_stats

    def _gen_user_features(self):
        self.log("生成用户特征...")

        user_stats = self.train_data.groupby('user_id').agg({
            'behavior_type': ['count', 'sum', 'mean'],
            'item_id': 'nunique',
            'is_weekend': 'mean'
        })
        user_stats.columns = ['u_total_actions', 'u_purchase_count', 'u_purchase_rate',
                              'u_unique_items', 'u_weekend_ratio']

        for w in BEST_WINDOWS:
            mask = self.window_masks[w]
            window_data = self.train_data[mask]
            if len(window_data) > 0:
                stats = window_data.groupby('user_id').agg({
                    'behavior_type': ['count', 'sum', 'mean']
                })
                stats.columns = [f'u_total_actions_{w}d', f'u_purchase_count_{w}d', f'u_purchase_rate_{w}d']
                user_stats = user_stats.merge(stats, left_index=True, right_index=True, how='left')
            else:
                user_stats[f'u_total_actions_{w}d'] = 0
                user_stats[f'u_purchase_count_{w}d'] = 0
                user_stats[f'u_purchase_rate_{w}d'] = 0

        u_slot = self.train_data.groupby('user_id')['time_slot'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 2
        )
        user_stats['u_preferred_time_slot'] = u_slot

        return user_stats.fillna(0)

    def _gen_ui_features(self):
        self.log("生成交互特征...")

        ui_stats = self.train_data.groupby(['user_id', 'item_id']).agg({
            'behavior_type': ['count', 'sum', 'mean'],
            'behavior_time': ['min', 'max']
        })
        ui_stats.columns = ['ui_total_actions', 'ui_has_purchased', 'ui_purchase_rate',
                            'ui_first_time', 'ui_last_time']

        bh = self.train_data.pivot_table(
            index=['user_id', 'item_id'], columns='behavior_type',
            values='behavior_time', aggfunc='count', fill_value=0
        )
        for b, name in [(1, 'view'), (2, 'fav'), (3, 'cart'), (4, 'purchase')]:
            ui_stats[f'ui_{name}_count'] = bh.get(b, 0)

        ui_stats['ui_has_carted'] = (ui_stats['ui_cart_count'] > 0).astype(int)
        ui_stats['ui_has_faved'] = (ui_stats['ui_fav_count'] > 0).astype(int)

        ui_stats['ui_hours_since_last_action'] = (
                                                         self.current_time - ui_stats['ui_last_time']
                                                 ).dt.total_seconds() / 3600
        ui_stats['ui_is_today'] = (ui_stats['ui_hours_since_last_action'] < 24).astype(int)
        ui_stats['ui_is_very_recent'] = (ui_stats['ui_hours_since_last_action'] < 1).astype(int)

        # 购买时间特征
        purchase_mask = self.train_data['behavior_type'] == 4
        if purchase_mask.any():
            purchase_times = self.train_data[purchase_mask].groupby(
                ['user_id', 'item_id']
            )['behavior_time'].min().rename('purchase_time')

            ui_stats = ui_stats.merge(purchase_times, left_index=True, right_index=True, how='left')
            ui_stats['ui_time_from_first_view_to_purchase'] = (
                                                                      ui_stats['purchase_time'] - ui_stats[
                                                                  'ui_first_time']
                                                              ).dt.total_seconds() / 3600
            ui_stats['ui_views_before_purchase'] = ui_stats.apply(
                lambda x: x['ui_view_count'] if pd.notna(x['purchase_time']) else 0, axis=1
            )
            ui_stats.drop('purchase_time', axis=1, inplace=True)
        else:
            ui_stats['ui_time_from_first_view_to_purchase'] = np.nan
            ui_stats['ui_views_before_purchase'] = 0

        days = (self.current_time - ui_stats['ui_first_time']).dt.total_seconds() / 86400 + 0.1
        ui_stats['ui_view_frequency'] = ui_stats['ui_view_count'] / days

        recent_3d = self.train_data[self.train_data['days_diff'] <= 3].groupby(['user_id', 'item_id']).size()
        ui_stats['ui_recent_action_concentration'] = (
                recent_3d.reindex(ui_stats.index).fillna(0) / ui_stats['ui_total_actions'].replace(0, np.nan)
        ).fillna(0)

        ui_stats['ui_sequence_length'] = (
                (ui_stats['ui_view_count'] > 0).astype(int) +
                (ui_stats['ui_fav_count'] > 0).astype(int) +
                (ui_stats['ui_cart_count'] > 0).astype(int) +
                (ui_stats['ui_purchase_count'] > 0).astype(int)
        )

        ui_stats['ui_has_fav_cart_pattern'] = ((ui_stats['ui_fav_count'] > 0) & (ui_stats['ui_cart_count'] > 0)).astype(
            int)
        ui_stats['ui_completed_funnel'] = (
                (ui_stats['ui_view_count'] > 0) & (ui_stats['ui_fav_count'] > 0) &
                (ui_stats['ui_cart_count'] > 0) & (ui_stats['ui_purchase_count'] > 0)
        ).astype(int)

        return ui_stats.fillna(0)

    def _gen_time_features_4h(self):
        self.log("生成时间匹配特征...")

        i_slot = self.train_data.groupby('item_id')['time_slot'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 4
        )

        u_slot = self.train_data.groupby('user_id')['time_slot'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 2
        )

        ui_pairs = self.train_data[['user_id', 'item_id']].drop_duplicates().set_index(['user_id', 'item_id'])
        ui_pairs['u_preferred_time_slot'] = u_slot.reindex(ui_pairs.index.get_level_values(0)).values
        ui_pairs['i_hot_time_slot'] = i_slot.reindex(ui_pairs.index.get_level_values(1)).values

        diff = abs(ui_pairs['u_preferred_time_slot'] - ui_pairs['i_hot_time_slot'])
        ui_pairs['ui_time_slot_match'] = (diff == 0).astype(int) * 1.0 + (diff == 1).astype(int) * 0.5

        u_weekend = self.train_data.groupby('user_id')['is_weekend'].mean()
        i_weekend = self.train_data.groupby('item_id')['is_weekend'].mean()
        ui_pairs['u_weekend_ratio'] = u_weekend.reindex(ui_pairs.index.get_level_values(0)).values
        ui_pairs['i_weekend_ratio'] = i_weekend.reindex(ui_pairs.index.get_level_values(1)).values
        ui_pairs['ui_weekend_match'] = 1 - abs(ui_pairs['u_weekend_ratio'] - ui_pairs['i_weekend_ratio'])

        return ui_pairs[['u_preferred_time_slot', 'i_hot_time_slot', 'ui_time_slot_match',
                         'u_weekend_ratio', 'i_weekend_ratio', 'ui_weekend_match']]

    def _gen_advanced_features(self, labels):
        self.log("生成高级特征...")

        advanced = labels[['user_id', 'item_id']].copy().set_index(['user_id', 'item_id'])

        item_viewers = self.train_data[self.train_data['behavior_type'] == 1].groupby('item_id')['user_id'].nunique()
        advanced['i_viewer_count'] = item_viewers.reindex(advanced.index.get_level_values(1)).fillna(0).values
        advanced['ui_is_only_viewer'] = (advanced['i_viewer_count'] == 1).astype(int)

        # 去掉双12特殊处理，改为通用时间衰减
        advanced['ui_recency_score'] = np.exp(
            -0.1 * self.train_data.groupby(['user_id', 'item_id'])['days_diff'].min().reindex(
                advanced.index).fillna(30).values)

        return advanced.fillna(0)

    def _calc_derived_features(self, df):
        self.log("计算衍生特征...")

        df['i_browse_to_buy_rate'] = (df['i_purchase_count'] + 1) / (df['i_browse_count'] + 10)
        df['i_cart_to_buy_rate'] = (df['i_purchase_count'] + 1) / (df['i_cart_count'] + 5)
        df['i_fav_to_buy_rate'] = (df['i_purchase_count'] + 1) / (df['i_fav_count'] + 5)

        total = df['i_browse_count'] + df['i_fav_count'] + df['i_cart_count'] + df['i_purchase_count'] + 0.1
        df['i_browse_ratio'] = df['i_browse_count'] / total
        df['i_fav_ratio'] = df['i_fav_count'] / total
        df['i_cart_ratio'] = df['i_cart_count'] / total
        df['i_purchase_ratio'] = df['i_purchase_count'] / total

        df['ui_attention_intensity'] = df['ui_total_actions'] / (
                df['u_total_actions'] / df['u_unique_items'].replace(0, 1) + 0.1)
        df['ui_is_impulse_buy_fast'] = (df['ui_time_from_first_view_to_purchase'] < 1).astype(int).fillna(0)
        df['ui_is_impulse_buy_no_depth'] = ((df['ui_view_count'] == 1) & (df['ui_has_purchased'] == 1)).astype(int)
        df['ui_popularity_preference_score'] = df['i_act_total'] * (
                df['ui_total_actions'] / df['u_total_actions'].replace(0, 1))
        df['ui_personalized_hot_score'] = df['i_act_total_7d'] * (df['ui_view_count'] + 1)
        df['ui_funnel_progress'] = df['ui_view_count'] * 0.1 + df['ui_fav_count'] * 0.2 + df['ui_cart_count'] * 0.3
        df['ui_cart_to_purchase_hours'] = df['ui_time_from_first_view_to_purchase'].fillna(0)
        df['ui_has_action_after_cart'] = (
                (df['ui_cart_count'] > 0) & (df['ui_view_count'] > df['ui_cart_count'])).astype(int)

        return df


class FeatureSelector:
    def __init__(self, features_df, label_col='label'):
        self.features_df = features_df
        self.label_col = label_col

        # 自动排除ID列、标签列和非数值列（如datetime）
        self.feature_cols = []
        for c in features_df.columns:
            if c in ['user_id', 'item_id', label_col]:
                continue
            # 只保留数值类型（int, float, bool）
            if pd.api.types.is_numeric_dtype(features_df[c]):
                self.feature_cols.append(c)
            else:
                print(f"  排除非数值列: {c} ({features_df[c].dtype})")

        print(f"可用数值特征: {len(self.feature_cols)}个")

    def calc_iv(self, feature, n_bins=10):
        try:
            df = self.features_df[[feature, self.label_col]].copy()
            df = df[df[feature].notna()]

            if df[feature].nunique() < 2:
                return 0

            # 确保是数值类型
            if not pd.api.types.is_numeric_dtype(df[feature]):
                return 0

            df['bin'] = pd.qcut(df[feature], q=min(n_bins, df[feature].nunique()),
                                duplicates='drop', labels=False)

            grouped = df.groupby('bin')[self.label_col].agg(['count', 'sum'])
            grouped['neg'] = grouped['count'] - grouped['sum']

            grouped['pos_rate'] = (grouped['sum'] + 0.5) / (grouped['sum'].sum() + 0.5)
            grouped['neg_rate'] = (grouped['neg'] + 0.5) / (grouped['neg'].sum() + 0.5)

            grouped['woe'] = np.log(grouped['pos_rate'] / grouped['neg_rate'])
            grouped['iv'] = (grouped['pos_rate'] - grouped['neg_rate']) * grouped['woe']

            return grouped['iv'].sum()
        except:
            return 0

    def screen_features(self, iv_threshold=0.02, corr_threshold=0.98, top_k=30):
        print(f"\n{'=' * 60}")
        print("阶段1: IV值筛选")
        print(f"{'=' * 60}")

        iv_scores = {}
        for feat in tqdm(self.feature_cols, desc="计算IV"):
            iv_scores[feat] = self.calc_iv(feat)

        iv_df = pd.DataFrame(list(iv_scores.items()), columns=['feature', 'iv'])
        iv_df = iv_df.sort_values('iv', ascending=False)

        selected = iv_df[iv_df['iv'] >= iv_threshold]['feature'].tolist()
        print(f"IV >= {iv_threshold}: 保留 {len(selected)}/{len(self.feature_cols)} 个特征")
        print(f"Top 10 IV:\n{iv_df.head(10).to_string(index=False)}")

        print(f"\n{'=' * 60}")
        print(f"阶段2: 相关性去冗余 (>{corr_threshold})")
        print(f"{'=' * 60}")

        if len(selected) > 1:
            corr_matrix = self.features_df[selected].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            to_drop = set()
            for col in upper.columns:
                high_corr = upper[col][upper[col] > corr_threshold].index.tolist()
                for corr_feat in high_corr:
                    if iv_scores[col] < iv_scores[corr_feat]:
                        to_drop.add(col)
                    else:
                        to_drop.add(corr_feat)

            selected = [f for f in selected if f not in to_drop]
            print(f"剔除 {len(to_drop)} 个, 剩余 {len(selected)} 个")

        print(f"\n{'=' * 60}")
        print("阶段3: LightGBM重要性")
        print(f"{'=' * 60}")

        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

        X = self.features_df[selected]
        y = self.features_df[self.label_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            random_state=42, verbose=-1, class_weight='balanced'
        )

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(10, verbose=False)])

        importance = pd.DataFrame({
            'feature': selected,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        final_features = importance.head(top_k)['feature'].tolist()
        print(f"Top {top_k}:\n", importance.head(top_k).to_string(index=False))

        return {
            'iv_scores': iv_df,
            'importance': importance,
            'final_features': final_features,
            'model': model
        }


def construct_labels_for_period(data, start_date, end_date, sample_ratio=3):
    """
    为指定时间段构造样本标签
    sample_ratio: 负采样比例（相对于正样本）
    """
    # 转换日期格式确保可比
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    mask = (data['date'] >= start_date) & (data['date'] <= end_date)
    period_data = data[mask]

    # 正样本：当天购买
    positives = period_data[period_data['behavior_type'] == 4][['user_id', 'item_id']].drop_duplicates()
    positives['label'] = 1
    print(f"  正样本: {len(positives)}")

    if len(positives) == 0:
        return pd.DataFrame(columns=['user_id', 'item_id', 'label'])

    # 负样本：当天有浏览但未购买
    all_pairs = period_data[period_data['behavior_type'] == 1][['user_id', 'item_id']].drop_duplicates()
    negatives = all_pairs.merge(positives, on=['user_id', 'item_id'], how='left', indicator=True)
    negatives = negatives[negatives['_merge'] == 'left_only'][['user_id', 'item_id']]
    negatives['label'] = 0

    # 限制负样本数量
    n_neg = min(len(negatives), int(len(positives) * sample_ratio))
    if n_neg > 0:
        negatives = negatives.sample(n=n_neg, random_state=42)

    print(f"  负样本: {len(negatives)}")
    return pd.concat([positives, negatives], ignore_index=True)


def main():
    print("=" * 80)
    print("全量数据特征工程（8:1:1时间序列划分）")
    print(f"总时间: 2014-11-18 至 2014-12-18 (31天)")
    print(f"训练集: 11-18 至 12-12 (25天, ~81%)")
    print(f"验证集: 12-13 至 12-15 (3天, ~10%)")
    print(f"测试集: 12-16 至 12-18 (3天, ~10%)")
    print("=" * 80)

    print("\n[1/5] 加载数据...")
    data = pd.read_parquet(DATA_PATH)

    # 数据预处理
    data['behavior_type'] = pd.to_numeric(data['behavior_type'], errors='coerce')
    if 'behavior_time' not in data.columns and 'datetime' in data.columns:
        data['behavior_time'] = pd.to_datetime(data['datetime'])
    if 'date' not in data.columns:
        data['date'] = pd.to_datetime(data['behavior_time']).dt.date
    else:
        data['date'] = pd.to_datetime(data['date']).dt.date

    print(f"全量数据: {len(data):,} 条")
    print(f"时间范围: {data['behavior_time'].min()} 至 {data['behavior_time'].max()}")

    # 分别构造三个数据集的样本和特征
    datasets = {}

    for split_name, (start_date, end_date) in DATE_RANGES.items():
        print(f"\n[2/5-{split_name}] 处理 {split_name} 集...")

        # 构造标签
        labels = construct_labels_for_period(
            data, start_date, end_date,
            sample_ratio=3 if split_name == 'train' else 10
        )

        if len(labels) == 0:
            print(f"  警告: {split_name} 无数据")
            continue

        # 特征工程：使用截止到该时间段开始前一天的数据（避免泄露）
        # 例如训练集标签是11-18至12-12的购买，特征用11-18之前的数据？不对
        # 应该用截止到start_date之前的数据？也不对，那样训练集就没数据了
        # 正确做法：对于时间序列，训练集用11-18至12-12的特征预测12-13的购买？也不对
        # 应该是：用历史数据预测未来，所以：
        # 训练集：用11-18至12-11的数据预测12-12的购买
        # 但这里start_date是11-18，所以调整逻辑：

        if split_name == 'train':
            # 训练集：用11-18至12-11的特征（25天历史）预测12-12的购买？不对，要预测整个区间
            # 实际上应该用截止到start_date之前的数据，但11-18是第一天
            # 修正：用整个训练期之前的数据，但训练期是第一天开始，所以用训练期内之前的数据
            feature_cutoff = datetime(2014, 12, 13)  # 不包含12-13，即用到12-12
        elif split_name == 'val':
            feature_cutoff = datetime(2014, 12, 16)  # 用到12-15
        else:
            feature_cutoff = datetime(2014, 12, 19)  # 用到12-18

        train_mask = data['behavior_time'] < feature_cutoff
        train_data = data[train_mask].copy()

        print(f"  特征计算截止: {feature_cutoff.date()}, 历史数据: {len(train_data):,} 条")

        # 生成特征
        engineer = OptimizedFeatureEngineer(train_data, feature_cutoff)
        features_df = engineer.generate_all_features(labels)

        datasets[split_name] = features_df

        # 保存原始特征（全量）
        raw_path = os.path.join(RESULTS_DIR, f'{split_name}_features_raw.parquet')
        features_df.to_parquet(raw_path)
        print(f"  保存原始特征: {raw_path}")

    # 特征选择（基于训练集）
    print(f"\n[3/5] 特征选择（基于训练集）...")
    train_df = datasets['train']

    selector = FeatureSelector(train_df)
    results = selector.screen_features(iv_threshold=0.01, top_k=30)

    selected_features = results['final_features']

    # 保存特征选择结果
    joblib.dump(results, os.path.join(RESULTS_DIR, 'feature_selection_results.pkl'))

    # 保存特征列表
    with open(os.path.join(RESULTS_DIR, 'selected_features_30.txt'), 'w') as f:
        for i, feat in enumerate(selected_features, 1):
            f.write(f"{i}. {feat}\n")

    print(f"\n[4/5] 筛选后特征保存...")
    # 对每个数据集保存筛选后的30个特征 + ID + label
    for split_name, df in datasets.items():
        selected_df = df[['user_id', 'item_id', 'label'] + selected_features]

        # 保存为parquet（推荐）和csv（兼容）
        parquet_path = os.path.join(RESULTS_DIR, f'{split_name}_selected_30.parquet')
        csv_path = os.path.join(RESULTS_DIR, f'{split_name}_selected_30.csv')

        selected_df.to_parquet(parquet_path)
        selected_df.to_csv(csv_path, index=False)

        print(f"  {split_name}: {len(selected_df):,} 样本, {len(selected_features)} 特征")
        print(f"    正样本率: {selected_df['label'].mean():.2%}")
        print(f"    保存至: {parquet_path}")

    # 保存特征说明文档
    print(f"\n[5/5] 生成特征文档...")
    doc_content = f"""# 全量数据特征筛选报告
时间范围: 2014-11-18 至 2014-12-18 (共31天)
划分方式: 时间序列 8:1:1
- 训练集: 11-18 至 12-12 (25天, ~81%)
- 验证集: 12-13 至 12-15 (3天, ~10%)
- 测试集: 12-16 至 12-18 (3天, ~10%)

筛选方法:
1. IV值筛选 (threshold=0.01)
2. 相关性去冗余 (threshold=0.98)
3. LightGBM重要性排序 (Top 30)

最终特征列表:
"""
    for i, feat in enumerate(selected_features, 1):
        doc_content += f"{i}. {feat}\n"

    with open(os.path.join(RESULTS_DIR, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(doc_content)

    print(f"\n✓ 全部完成！结果保存在: {RESULTS_DIR}")
    print(f"  可直接用于逻辑回归的文件:")
    print(f"  - {RESULTS_DIR}/train_selected_30.parquet")
    print(f"  - {RESULTS_DIR}/val_selected_30.parquet")
    print(f"  - {RESULTS_DIR}/test_selected_30.parquet")


if __name__ == "__main__":
    main()