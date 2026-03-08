"""
高性能特征工程系统（数据格式修复版）
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

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

DATA_PATH = os.path.join(project_root, 'data', 'processed', 'user_action_processed.parquet')
RESULTS_DIR = os.path.join(project_root, 'results', 'feature_engineering_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)

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
        print(f"[{time.time() - self.start_time:.1f}s] {msg}")

    def generate_all_features(self, labels):
        if len(labels) == 0:
            raise ValueError("标签为空")

        self.log("开始特征工程...")

        item_features = self._gen_item_features()
        user_features = self._gen_user_features()
        ui_features = self._gen_ui_features()
        time_features = self._gen_time_features_4h()
        advanced_features = self._gen_advanced_features(labels)

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

        recent = self.train_data[self.train_data['days_diff'] <= 7].groupby('item_id').size()
        prev = self.train_data[(self.train_data['days_diff'] > 7) &
                               (self.train_data['days_diff'] <= 14)].groupby('item_id').size()
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

        # 购买时间特征（向量化）
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
        self.log("生成4小时分箱时间特征...")

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

        if 'category_id' in self.train_data.columns:
            cat_hot = self.train_data.groupby('category_id')['behavior_type'].count()
            item_cat = self.train_data.groupby('item_id')['category_id'].first()
            advanced['i_category_id'] = item_cat.reindex(advanced.index.get_level_values(1)).values
            advanced['i_similar_items_popularity'] = cat_hot.reindex(advanced['i_category_id']).fillna(0).values
        else:
            advanced['i_similar_items_popularity'] = 0

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
        self.feature_cols = [c for c in features_df.columns
                             if c not in ['user_id', 'item_id', label_col]]

    def calc_iv(self, feature, n_bins=10):
        try:
            df = self.features_df[[feature, self.label_col]].copy()
            df = df[df[feature].notna()]

            if df[feature].nunique() < 2:
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

    def screen_features(self, iv_threshold=0.02, corr_threshold=0.95, top_k=30):
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
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            random_state=42, verbose=-1
        )

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(10, verbose=False)])

        importance = pd.DataFrame({
            'feature': selected,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("Top 15:\n", importance.head(15).to_string(index=False))

        return {
            'iv_scores': iv_df,
            'importance': importance,
            'final_features': importance.head(top_k)['feature'].tolist()
        }, model


def main():
    print("=" * 80)
    print("高性能特征工程系统 (数据修复版)")
    print(f"窗口: {BEST_WINDOWS}")
    print("=" * 80)

    print("\n[1/4] 加载数据...")
    data = pd.read_parquet(DATA_PATH)

    # ===== 关键修复 =====
    # 1. 强制转换behavior_type为整数（防止是字符串'1'而不是数字1）
    data['behavior_type'] = pd.to_numeric(data['behavior_type'], errors='coerce')

    # 2. 使用已有的behavior_time或从datetime解析
    if 'behavior_time' not in data.columns and 'datetime' in data.columns:
        data['behavior_time'] = pd.to_datetime(data['datetime'])
    elif 'datetime' in data.columns:
        data['behavior_time'] = pd.to_datetime(data['datetime'])

    # 3. 使用已有的date列，或从behavior_time生成
    if 'date' not in data.columns:
        data['date'] = data['behavior_time'].dt.date
    else:
        # 确保date是date对象而不是字符串
        if isinstance(data['date'].iloc[0], str):
            data['date'] = pd.to_datetime(data['date']).dt.date

    print(f"时间范围: {data['behavior_time'].min()} 到 {data['behavior_time'].max()}")
    print(f"behavior_type分布:\n{data['behavior_type'].value_counts().sort_index()}")

    # 4. 找有购买的日期
    purchase_dates = data[data['behavior_type'] == 4]['date'].unique()
    print(f"\n有购买的日期数: {len(purchase_dates)}")
    if len(purchase_dates) > 0:
        print(f"最近5天: {sorted(purchase_dates)[-5:]}")
        test_date = sorted(purchase_dates)[-1]  # 使用最后一个有购买的日期
    else:
        raise ValueError("数据中没有购买行为(behavior_type=4)！")

    print(f"\n[2/4] 时间划分...")
    # 训练集：test_date之前14天
    test_datetime = datetime.combine(test_date, datetime.min.time())
    train_end = test_datetime
    train_start = train_end - timedelta(days=14)

    # 使用date列进行过滤（更可靠）
    train_data = data[(data['behavior_time'] >= train_start) &
                      (data['behavior_time'] < train_end)].copy()
    test_data = data[(data['date'] == test_date) &
                     (data['behavior_type'] == 4)].copy()

    print(f"训练集: {train_start.date()} 至 {(train_end - timedelta(seconds=1)).date()} ({len(train_data):,}条)")
    print(f"测试集: {test_date} 购买 ({len(test_data):,}条)")

    if len(test_data) == 0:
        raise ValueError(f"{test_date}没有购买数据！")

    # 5. 构造样本
    print("\n[3/4] 构造样本...")
    positives = test_data[['user_id', 'item_id']].drop_duplicates()
    positives['label'] = 1
    print(f"正样本: {len(positives)}")

    if len(positives) > 5000:
        positives = positives.sample(5000, random_state=42)

    # 生成负样本
    selected_users = positives['user_id'].unique()
    train_browsed = train_data[train_data['behavior_type'] == 1].groupby('user_id')['item_id'].apply(set).to_dict()
    train_purchased = train_data[train_data['behavior_type'] == 4].groupby('user_id')['item_id'].apply(set).to_dict()

    negatives = []
    for uid in tqdm(selected_users, desc="配对负样本"):
        pos_items = set(positives[positives['user_id'] == uid]['item_id'])
        browsed = train_browsed.get(uid, set())
        purchased = train_purchased.get(uid, set())
        candidates = list(browsed - purchased - pos_items)

        if len(candidates) > 0:
            n_neg = min(len(candidates), len(pos_items) * 3)
            if n_neg > 0:
                sampled = np.random.choice(candidates, size=n_neg, replace=False)
                negatives.extend([{'user_id': uid, 'item_id': iid, 'label': 0} for iid in sampled])

    negatives_df = pd.DataFrame(negatives) if negatives else pd.DataFrame(columns=['user_id', 'item_id', 'label'])
    labels = pd.concat([positives, negatives_df], ignore_index=True)

    print(f"负样本: {len(negatives_df)}, 总计: {len(labels)}")

    # 6. 特征工程
    print("\n[4/4] 特征工程...")
    start_time = time.time()
    engineer = OptimizedFeatureEngineer(train_data, train_end)
    features_df = engineer.generate_all_features(labels)

    # 7. 筛选
    print("\n[5/5] 特征筛选...")
    selector = FeatureSelector(features_df)
    results, model = selector.screen_features(iv_threshold=0.01, top_k=30)

    # 输出
    print(f"\n{'=' * 80}")
    print("最终Top 30特征:")
    for i, feat in enumerate(results['final_features'], 1):
        print(f"{i:2d}. {feat}")

    output_df = features_df[results['final_features'] + ['user_id', 'item_id', 'label']]
    output_path = os.path.join(RESULTS_DIR, 'selected_features.csv')
    output_df.to_csv(output_path, index=False)
    print(f"\n保存至: {output_path}")
    print(f"总时间: {time.time() - start_time:.1f}秒")


if __name__ == "__main__":
    main()