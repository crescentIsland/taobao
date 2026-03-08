"""
完整数据时间窗口验证 - 高性能分块优化版（完整修正版）
针对1200万+大数据集优化，控制内存使用
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
from collections import defaultdict

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

DATA_PATH = os.path.join(project_root, 'data', 'processed', 'user_action_processed.parquet')
RESULTS_DIR = os.path.join(project_root, 'results', 'full_data_chunked_fixed')
os.makedirs(RESULTS_DIR, exist_ok=True)


class ChunkedTimeWindowExperiment:
    """分块处理大数据集的时间窗口实验（完整修正版）"""

    def __init__(self, data_path, chunk_size=500000):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.data = None
        self.results = []
        self.start_time = time.time()
        self.stats = {}

    def log(self, message):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.1f}s] {message}")

    def load_data(self):
        """标准加载"""
        self.log("加载数据...")
        self.data = pd.read_parquet(self.data_path)
        self.log(f"数据加载完成: {len(self.data):,} 条记录")

        # 数据类型优化
        self.data['behavior_type'] = pd.to_numeric(self.data['behavior_type'], errors='coerce')
        self.data = self.data[self.data['behavior_type'].notnull()]
        self.data['behavior_type'] = self.data['behavior_type'].astype(np.int8)

        if 'datetime' in self.data.columns:
            self.data['behavior_time'] = pd.to_datetime(self.data['datetime'])

        self.data = self.data.sort_values(['user_id', 'behavior_time'])

        # 记录基础统计
        self.stats['total_records'] = len(self.data)
        self.stats['unique_users'] = self.data['user_id'].nunique()
        self.stats['unique_items'] = self.data['item_id'].nunique()
        self.stats['unique_pairs'] = self.data[['user_id', 'item_id']].drop_duplicates().shape[0]

        self.log(f"基础统计:")
        self.log(f"  唯一用户数: {self.stats['unique_users']:,}")
        self.log(f"  唯一商品数: {self.stats['unique_items']:,}")
        self.log(f"  唯一(user,item)对: {self.stats['unique_pairs']:,}")
        self.log(f"时间范围: {self.data['behavior_time'].min()} 到 {self.data['behavior_time'].max()}")

        return True

    def create_experiment_with_gap(self):
        """创建实验设置（严格分离训练集和测试集）"""
        self.log("创建实验设置（严格分离版）...")

        max_date = self.data['behavior_time'].max()
        max_date_date = max_date.date()  # 应该是2014-12-18

        # 测试集：最后一天（12-18）全天 [00:00:00, 23:59:59]
        test_start = datetime.combine(max_date_date, datetime.min.time())  # 2014-12-18 00:00:00
        test_end = test_start + timedelta(days=1)  # 2014-12-19 00:00:00（不包含）

        # 训练集：test_start之前14天，到17号结束（不包含18号任何数据）
        train_days = 14
        train_end = test_start  # 2014-12-18 00:00:00（不包含）
        train_start = train_end - timedelta(days=train_days)  # 2014-12-04 00:00:00

        print(f"\n实验设置:")
        print(f"  训练集: {train_start.date()} 到 {(train_end - timedelta(seconds=1)).date()} ({train_days}天)")
        print(f"  测试集: {test_start.date()} (单独一天，与训练集严格无重叠)")
        print(f"  分割点: {train_end}（训练集<此时间，测试集>=此时间）")

        # 向量化过滤（严格左闭右开）
        train_mask = (self.data['behavior_time'] >= train_start) & (self.data['behavior_time'] < train_end)
        train_data = self.data[train_mask].copy()

        test_mask = (self.data['behavior_time'] >= test_start) & (self.data['behavior_time'] < test_end) & (
                    self.data['behavior_type'] == 4)
        test_data = self.data[test_mask].copy()

        # 验证：确保训练集没有18号数据，测试集只有18号数据
        if len(train_data) > 0:
            train_max_date = train_data['behavior_time'].max().date()
            assert train_max_date < test_start.date(), f"训练集包含测试集日期！最大日期：{train_max_date}"
            self.log(f"  [验证通过] 训练集最大日期: {train_max_date}")

        if len(test_data) > 0:
            test_min_date = test_data['behavior_time'].min().date()
            test_max_date = test_data['behavior_time'].max().date()
            assert test_min_date == test_start.date(), f"测试集日期错误！最小日期：{test_min_date}"
            assert test_max_date == test_start.date(), f"测试集包含多天！最大日期：{test_max_date}"
            self.log(f"  [验证通过] 测试集日期范围: {test_min_date} 到 {test_max_date}")

        # 记录统计
        self.stats['train_records'] = len(train_data)
        self.stats['test_purchases'] = len(test_data)
        self.stats['train_unique_users'] = train_data['user_id'].nunique()
        self.stats['train_unique_items'] = train_data['item_id'].nunique()

        # 立即释放原始数据内存
        del self.data
        gc.collect()

        print(f"\n数据统计:")
        print(f"  训练数据: {len(train_data):,} 条记录")
        print(f"  训练集唯一用户: {self.stats['train_unique_users']:,}")
        print(f"  训练集唯一商品: {self.stats['train_unique_items']:,}")
        print(f"  测试集购买行为: {len(test_data):,}")

        return train_data, test_data, test_start

    def analyze_candidate_pool(self, train_data, test_data):
        """分析候选样本池（购买 或 交互>=2次）"""
        self.log("分析候选样本池...")

        # 1. 正样本池：测试集中的购买对（12-18的购买）
        positive_pool = test_data[['user_id', 'item_id']].drop_duplicates()
        positive_pool['label'] = 1
        self.stats['positive_pool_size'] = len(positive_pool)

        # 2. 计算训练集中每个(user, item)的交互次数
        self.log("  统计训练集交互频次...")
        interaction_counts = train_data.groupby(['user_id', 'item_id']).size().reset_index(name='interaction_count')

        # 3. 负样本候选池：训练集中交互>=2次但没有购买的对
        # 先找出训练集中的购买（用于排除）
        train_purchases = train_data[train_data['behavior_type'] == 4][['user_id', 'item_id']].drop_duplicates()
        train_purchases['is_purchase'] = 1

        # 合并
        candidate_negatives = interaction_counts.merge(
            train_purchases, on=['user_id', 'item_id'], how='left'
        )
        # 筛选：交互>=2次且未购买
        candidate_negatives = candidate_negatives[
            (candidate_negatives['interaction_count'] >= 2) &
            (candidate_negatives['is_purchase'].isna())
            ][['user_id', 'item_id', 'interaction_count']]

        self.stats['negative_pool_size'] = len(candidate_negatives)

        # 4. 合并候选池
        candidate_pool = pd.concat([
            positive_pool[['user_id', 'item_id', 'label']],
            candidate_negatives[['user_id', 'item_id']].assign(label=0)
        ], ignore_index=True)

        self.stats['total_candidate_pool'] = len(candidate_pool)

        self.log(f"候选样本池统计:")
        self.log(f"  正样本池（测试集购买）: {self.stats['positive_pool_size']:,}")
        self.log(f"  负样本池（训练集交互>=2次未购买）: {self.stats['negative_pool_size']:,}")
        self.log(f"  总候选池: {self.stats['total_candidate_pool']:,}")

        return candidate_pool, positive_pool, candidate_negatives

    def stratified_sampling(self, positive_pool, candidate_negatives, target_total=12000, neg_ratio=3):
        """分层抽样构造最终训练样本（取消max_users限制）"""
        self.log("分层抽样构造训练样本...")

        # 目标：正样本全量（或上限5000），负样本按配比
        n_pos_available = len(positive_pool)
        n_pos = min(n_pos_available, 5000)  # 正样本上限5000，但不再限制用户

        # 如果正样本太多，分层抽样保留用户多样性
        if n_pos < n_pos_available:
            self.log(f"  正样本过多({n_pos_available})，分层抽样保留{n_pos}个")
            # 按用户分层：优先保留有多个购买的用户（高价值用户）
            user_counts = positive_pool['user_id'].value_counts()
            positive_pool['user_purchase_freq'] = positive_pool['user_id'].map(user_counts)
            # 先按用户频次排序，再在每个用户内保留购买
            positive_sampled = positive_pool.sort_values(['user_purchase_freq', 'user_id'], ascending=False).head(n_pos)
        else:
            positive_sampled = positive_pool.copy()

        # 需要的负样本数
        n_neg_needed = n_pos * neg_ratio

        # 为每个正样本用户配负样本（不再限制max_users，用全部相关用户）
        selected_users = positive_sampled['user_id'].unique()
        self.stats['selected_users_for_sampling'] = len(selected_users)
        self.log(f"  涉及用户: {len(selected_users):,}人")

        negatives = []

        for user_id in tqdm(selected_users, desc="配对负样本", unit="用户"):
            # 该用户的正样本商品
            user_pos_items = set(positive_sampled[positive_sampled['user_id'] == user_id]['item_id'])
            n_pos_user = len(user_pos_items)

            # 该用户的候选负样本（交互>=2次未购买）
            user_neg_candidates = candidate_negatives[candidate_negatives['user_id'] == user_id]

            if len(user_neg_candidates) > 0:
                # 需要的负样本数（1:3比例，但不低于1个）
                n_neg_user = min(len(user_neg_candidates), max(1, n_pos_user * neg_ratio))

                # 分层：优先选择交互次数高的（更强的负信号）
                user_neg_candidates = user_neg_candidates.sort_values('interaction_count', ascending=False)
                sampled_neg = user_neg_candidates.head(int(n_neg_user))
                negatives.append(sampled_neg[['user_id', 'item_id']].assign(label=0))

        negatives_df = pd.concat(negatives, ignore_index=True) if negatives else pd.DataFrame(
            columns=['user_id', 'item_id', 'label'])

        # 合并
        labels = pd.concat([
            positive_sampled[['user_id', 'item_id', 'label']],
            negatives_df
        ], ignore_index=True)

        self.stats['final_positive_samples'] = len(positive_sampled)
        self.stats['final_negative_samples'] = len(negatives_df)
        self.stats['final_total_samples'] = len(labels)
        self.stats['final_pos_neg_ratio'] = len(negatives_df) / len(positive_sampled) if len(
            positive_sampled) > 0 else 0

        self.log(f"最终训练样本:")
        self.log(f"  正样本: {self.stats['final_positive_samples']:,}")
        self.log(f"  负样本: {self.stats['final_negative_samples']:,}")
        self.log(f"  总样本: {self.stats['final_total_samples']:,}")
        self.log(f"  正负比例: 1:{self.stats['final_pos_neg_ratio']:.1f}")

        return labels

    def precompute_user_features_chunked(self, train_data, windows, current_time, relevant_users):
        """分块预计算用户特征（只计算相关用户）"""
        self.log("分块预计算用户特征...")

        # 关键优化：只保留会出现在样本中的用户
        train_data_filtered = train_data[train_data['user_id'].isin(relevant_users)].copy()
        self.log(f"  过滤后数据: {len(train_data_filtered):,} 条 (原{len(train_data):,})")
        self.log(f"  相关用户数: {len(relevant_users):,}")

        user_features = {}
        total_chunks = (len(train_data_filtered) + self.chunk_size - 1) // self.chunk_size

        train_data_filtered['days_diff'] = (current_time - train_data_filtered[
            'behavior_time']).dt.total_seconds() / 86400

        for i in tqdm(range(0, len(train_data_filtered), self.chunk_size), desc="用户特征分块", total=total_chunks):
            chunk = train_data_filtered.iloc[i:i + self.chunk_size].copy()

            for user_id, group in chunk.groupby('user_id'):
                if user_id not in user_features:
                    user_features[user_id] = {}

                for window in windows:
                    mask = group['days_diff'] <= window
                    window_data = group[mask]

                    if len(window_data) > 0:
                        user_features[user_id][f'u_total_{window}d'] = len(window_data)
                        behavior_counts = window_data['behavior_type'].value_counts()
                        for btype in [1, 2, 3, 4]:
                            user_features[user_id][f'u_type{btype}_{window}d'] = behavior_counts.get(btype, 0)
                    else:
                        user_features[user_id][f'u_total_{window}d'] = 0
                        for btype in [1, 2, 3, 4]:
                            user_features[user_id][f'u_type{btype}_{window}d'] = 0

            del chunk
            if i % (self.chunk_size * 4) == 0:
                gc.collect()

        del train_data_filtered
        gc.collect()

        return user_features

    def precompute_item_features_chunked(self, train_data, windows, current_time, relevant_items):
        """分块预计算商品特征（只计算相关商品）- 核心优化"""
        self.log("分块预计算商品特征（优化版）...")

        # 关键优化：只保留样本中出现的商品（从可能的几万降到几百/几千）
        train_data_filtered = train_data[train_data['item_id'].isin(relevant_items)].copy()
        self.log(f"  过滤后数据: {len(train_data_filtered):,} 条 (原{len(train_data):,})")
        self.log(f"  相关商品数: {len(relevant_items):,}")

        item_features = {}
        total_chunks = (len(train_data_filtered) + self.chunk_size - 1) // self.chunk_size

        train_data_filtered['days_diff'] = (current_time - train_data_filtered[
            'behavior_time']).dt.total_seconds() / 86400

        for i in tqdm(range(0, len(train_data_filtered), self.chunk_size), desc="商品特征分块", total=total_chunks):
            chunk = train_data_filtered.iloc[i:i + self.chunk_size].copy()

            for item_id, group in chunk.groupby('item_id'):
                if item_id not in item_features:
                    item_features[item_id] = {}

                for window in windows:
                    mask = group['days_diff'] <= window
                    window_data = group[mask]

                    if len(window_data) > 0:
                        item_features[item_id][f'i_total_{window}d'] = len(window_data)
                    else:
                        item_features[item_id][f'i_total_{window}d'] = 0

            del chunk
            if i % (self.chunk_size * 4) == 0:
                gc.collect()

        del train_data_filtered
        gc.collect()

        return item_features

    def generate_features_vectorized(self, labels, user_features, item_features, config_name):
        """向量化特征生成"""
        self.log(f"生成特征 - {config_name} (向量化)")

        user_feat_df = pd.DataFrame.from_dict(user_features, orient='index')
        item_feat_df = pd.DataFrame.from_dict(item_features, orient='index')

        features_df = labels.merge(
            user_feat_df, left_on='user_id', right_index=True, how='left'
        ).merge(
            item_feat_df, left_on='item_id', right_index=True, how='left'
        ).fillna(0)

        del user_feat_df, item_feat_df
        gc.collect()

        self.log(f"特征生成完成: {features_df.shape[0]} 样本 x {features_df.shape[1] - 3} 特征")
        return features_df

    def train_and_evaluate(self, features_df, config_name):
        """训练和评估"""
        self.log(f"训练模型 - {config_name}")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        feature_cols = [col for col in features_df.columns if col not in ['user_id', 'item_id', 'label']]

        if len(feature_cols) == 0:
            self.log("错误: 没有特征!")
            return 0, pd.DataFrame()

        X = features_df[feature_cols]
        y = features_df['label']

        if len(np.unique(y)) < 2:
            self.log("错误: 只有一个类别!")
            return 0, pd.DataFrame()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        self.log(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        try:
            train_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - train_start

            y_pred = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred)

            self.log(f"  训练用时: {train_time:.1f}秒")
            self.log(f"  AUC分数: {auc_score:.4f}")

            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            return auc_score, importance_df

        except Exception as e:
            self.log(f"  训练失败: {e}")
            return 0, pd.DataFrame()

    def run(self):
        """运行完整实验"""
        print("\n" + "=" * 80)
        print("开始分块优化版实验（完整修正版）")
        print(f"分块大小: {self.chunk_size:,}")
        print("=" * 80)

        # 1. 加载数据
        if not self.load_data():
            return

        # 2. 创建实验设置（严格分离训练集12-04~12-17和测试集12-18）
        train_data, test_data, test_date = self.create_experiment_with_gap()

        # 3. 分析候选样本池
        candidate_pool, positive_pool, candidate_negatives = self.analyze_candidate_pool(train_data, test_data)

        # 4. 分层抽样构造训练样本（取消max_users限制）
        labels = self.stratified_sampling(positive_pool, candidate_negatives, target_total=12000, neg_ratio=3)

        if len(labels) < 100:
            self.log("错误: 标签太少!")
            return

        # 5. 窗口配置测试
        window_configs = {
            'traditional': [1, 3, 7, 14, 21],
            'fine_grained': [1, 2, 3, 4, 5, 6, 7],
            'geometric': [1, 2, 4, 8, 16],
            'weekly': [1, 7, 14, 21, 28],
            'mixed': [1, 3, 7, 14, 30]
        }

        print(f"\n测试 {len(window_configs)} 种窗口配置")

        all_windows = sorted(set([w for windows in window_configs.values() for w in windows]))
        self.log(f"预计算所有窗口: {all_windows}")

        # 关键：提前获取相关用户和商品集合（用于加速预计算）
        relevant_users = set(labels['user_id'].unique())
        relevant_items = set(labels['item_id'].unique())
        self.log(f"相关用户: {len(relevant_users):,}, 相关商品: {len(relevant_items):,}")

        # 6. 预计算特征（优化版：只计算相关用户/商品）
        user_features = self.precompute_user_features_chunked(train_data, all_windows, test_date, relevant_users)
        item_features = self.precompute_item_features_chunked(train_data, all_windows, test_date, relevant_items)

        # 释放训练数据
        del train_data
        gc.collect()

        print("\n" + "-" * 80)

        # 7. 逐个测试配置
        for config_name, windows in window_configs.items():
            config_start = time.time()
            print(f"\n处理配置: {config_name}")
            print(f"窗口: {windows}")
            print("-" * 40)

            # 选择需要的窗口特征
            window_set = set(windows)
            user_feat_subset = {
                uid: {k: v for k, v in feats.items() if any(f'_{w}d' in k for w in window_set)}
                for uid, feats in user_features.items()
            }
            item_feat_subset = {
                iid: {k: v for k, v in feats.items() if any(f'_{w}d' in k for w in window_set)}
                for iid, feats in item_features.items()
            }

            features_df = self.generate_features_vectorized(
                labels, user_feat_subset, item_feat_subset, config_name
            )

            if features_df.empty:
                print(f"跳过 {config_name}")
                continue

            auc_score, importance_df = self.train_and_evaluate(features_df, config_name)

            if auc_score > 0:
                self.results.append({
                    'config': config_name,
                    'windows': str(windows),
                    'auc_score': auc_score,
                    'num_features': features_df.shape[1] - 3,
                    'num_samples': features_df.shape[0],
                    'config_time': time.time() - config_start
                })

                if not importance_df.empty:
                    importance_path = os.path.join(RESULTS_DIR, f'importance_{config_name}.csv')
                    importance_df.to_csv(importance_path, index=False)

                    window_features = importance_df[importance_df['feature'].str.contains('_total_')]
                    if not window_features.empty:
                        print(f"窗口特征重要性 (Top 5):")
                        for _, row in window_features.head(5).iterrows():
                            print(f"  {row['feature']}: {row['importance']:.4f}")

            del features_df, user_feat_subset, item_feat_subset
            gc.collect()

            print(f"配置完成，用时: {time.time() - config_start:.1f}秒")

        # 8. 分析结果
        self.analyze_results()

        # 9. 输出完整统计
        self.print_final_stats()

        print(f"\n总运行时间: {time.time() - self.start_time:.1f}秒")
        print("=" * 80)

    def analyze_results(self):
        """分析结果"""
        if not self.results:
            print("没有结果!")
            return

        results_df = pd.DataFrame(self.results).sort_values('auc_score', ascending=False)
        results_path = os.path.join(RESULTS_DIR, 'experiment_results.csv')
        results_df.to_csv(results_path, index=False)

        print("\n" + "=" * 80)
        print("实验结果汇总")
        print("=" * 80)

        print("\n配置排名:")
        for idx, row in results_df.iterrows():
            print(f"{idx + 1:2d}. {row['config']:15s} AUC={row['auc_score']:.4f} "
                  f"特征={row['num_features']:3d} 样本={row['num_samples']:5d} "
                  f"用时={row['config_time']:.1f}s")

    def print_final_stats(self):
        """输出完整统计报告"""
        print("\n" + "=" * 80)
        print("完整数据统计报告")
        print("=" * 80)

        stats_order = [
            ('total_records', '原始数据总记录数'),
            ('unique_users', '唯一用户数'),
            ('unique_items', '唯一商品数'),
            ('unique_pairs', '唯一(user,item)对'),
            ('train_records', '训练集记录数(12-04至12-17)'),
            ('train_unique_users', '训练集唯一用户'),
            ('train_unique_items', '训练集唯一商品'),
            ('test_purchases', '测试集购买行为数(12-18)'),
            ('positive_pool_size', '正样本池（测试集购买对）'),
            ('negative_pool_size', '负样本池（训练集交互>=2未购买）'),
            ('total_candidate_pool', '总候选样本池'),
            ('selected_users_for_sampling', '抽样涉及用户数'),
            ('final_positive_samples', '最终正样本数'),
            ('final_negative_samples', '最终负样本数'),
            ('final_total_samples', '最终训练样本总数'),
        ]

        for key, desc in stats_order:
            if key in self.stats:
                value = self.stats[key]
                if isinstance(value, float):
                    print(f"{desc}: {value:,.2f}")
                else:
                    print(f"{desc}: {value:,}")

        if 'final_pos_neg_ratio' in self.stats:
            print(f"正负样本比例: 1:{self.stats['final_pos_neg_ratio']:.2f}")


def main():
    print("分块优化时间窗口验证实验（完整修正版）")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    experiment = ChunkedTimeWindowExperiment(DATA_PATH, chunk_size=500000)
    experiment.run()

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()