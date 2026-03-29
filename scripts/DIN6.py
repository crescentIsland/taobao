"""
商品冷启动MLP - 纯统计特征版 (修复Config属性)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import warnings
import json
from collections import defaultdict

warnings.filterwarnings('ignore')


class Config:
    DATA_PATH = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet'
    SAVE_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\mlp_stat_only'
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_START = '2014-11-19'
    TEST_END = '2014-12-18'

    TRAIN_ITEM_RATIO = 0.8
    VAL_ITEM_RATIO = 0.1

    # 关键修复：添加缺失的属性
    NEGATIVE_RATIO = 1.0

    HIDDEN_DIMS = [64, 32, 16]
    DROPOUT_RATE = 0.5
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5
    BATCH_SIZE = 512
    EPOCHS = 100
    LR = 0.001

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StatOnlyBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_samples(self):
        print(f"[INFO] 加载数据: {self.cfg.DATA_PATH}")
        df = pd.read_parquet(self.cfg.DATA_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')

        print(f"      原始数据: {len(df):,}条")

        # 步骤1：商品划分
        print("[1/4] 商品冷启动划分...")
        target_df = df[(df['date'] >= self.cfg.TRAIN_START) & (df['date'] <= self.cfg.TEST_END)]
        purchased_items = target_df[target_df['behavior_type'] == '4']['item_id'].unique()
        purchased_items = [int(x) for x in purchased_items]

        np.random.seed(self.cfg.SEED)
        np.random.shuffle(purchased_items)

        n_items = len(purchased_items)
        n_train = int(n_items * self.cfg.TRAIN_ITEM_RATIO)
        n_val = int(n_items * self.cfg.VAL_ITEM_RATIO)

        train_items = set(purchased_items[:n_train])
        val_items = set(purchased_items[n_train:n_train + n_val])
        test_items = set(purchased_items[n_train + n_val:])

        print(f"      训练商品: {len(train_items):,} | 验证商品: {len(val_items):,} | 测试商品: {len(test_items):,}")

        # 步骤2：预计算统计（只用训练集，防止泄露）
        print("[2/4] 预计算统计...")
        train_df = df[df['item_id'].isin(train_items)]

        # 类别统计
        cate_stats = {}
        cate_behaviors = train_df.groupby(['item_category', 'behavior_type']).size().unstack(fill_value=0)
        for cate in cate_behaviors.index:
            views = cate_behaviors.loc[cate].get('1', 0)
            carts = cate_behaviors.loc[cate].get('3', 0)
            buys = cate_behaviors.loc[cate].get('4', 0)
            cate_stats[int(cate)] = {
                'cate_view': int(views),
                'cate_cart': int(carts),
                'cate_buy': int(buys),
                'cate_ctr': float(buys / (views + 1e-8)),
                'cate_cvr': float(buys / (carts + 1e-8)) if carts > 0 else 0
            }

        # 用户统计（购买力等）
        user_stats = {}
        user_behaviors = train_df.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
        for uid in user_behaviors.index:
            views = user_behaviors.loc[uid].get('1', 0)
            buys = user_behaviors.loc[uid].get('4', 0)
            user_stats[int(uid)] = {
                'user_total_buy': int(buys),
                'user_total_view': int(views),
                'user_buy_rate': float(buys / (views + 1e-8))
            }

        # 步骤3：构建索引
        print("[3/4] 构建索引...")
        user_histories = self._build_user_index(df)

        # 步骤4：构造样本（纯统计特征，无ID）
        print("[4/4] 构造样本（纯统计特征）...")
        train_samples, val_samples, test_samples = [], [], []

        purchases = target_df[target_df['behavior_type'] == '4'].sort_values('datetime')

        for _, row in tqdm(purchases.iterrows(), total=len(purchases), desc="构造"):
            user_id = int(row['user_id'])
            target_item = int(row['item_id'])
            target_cate = int(row['item_category'])
            cutoff_time = row['datetime']

            if target_item in train_items:
                target_list = train_samples
                allowed_items = train_items
            elif target_item in val_items:
                target_list = val_samples
                allowed_items = val_items
            elif target_item in test_items:
                target_list = test_samples
                allowed_items = test_items
            else:
                continue

            user_hist = self._get_user_history_before(user_histories, user_id, cutoff_time)
            if len(user_hist) == 0:
                continue

            # 用户近期统计（动态特征）
            recent_stats = self._fast_compute_stats(user_hist)

            # 用户全局统计（静态特征）
            user_global = user_stats.get(user_id, {'user_total_buy': 0, 'user_total_view': 0, 'user_buy_rate': 0})

            # 类别统计（商品侧特征）
            cate_feat = cate_stats.get(target_cate,
                                       {'cate_view': 0, 'cate_cart': 0, 'cate_buy': 0, 'cate_ctr': 0, 'cate_cvr': 0})

            # 时间特征
            hour = int(row['datetime'].hour)
            date_type = self._get_date_type(row['date'])

            # 组合所有数值特征（16维，无ID）
            sample = [
                # 标签
                1,  # label
                # 用户动态行为（近期）
                recent_stats['n_view'], recent_stats['n_fav'], recent_stats['n_cart'], recent_stats['n_buy'],
                recent_stats['len'], recent_stats['unique_items'], recent_stats['unique_cates'],
                # 用户静态画像（全局）
                user_global['user_total_buy'], user_global['user_total_view'], user_global['user_buy_rate'],
                # 商品/类别侧特征
                cate_feat['cate_view'], cate_feat['cate_cart'], cate_feat['cate_buy'],
                cate_feat['cate_ctr'], cate_feat['cate_cvr'],
                # 上下文
                hour, date_type
            ]
            target_list.append(sample)

            # 负采样
            hist_items = set(int(h[1]) for h in user_hist)
            neg_candidates = list((hist_items - {target_item}) & allowed_items)

            if neg_candidates:
                n_neg = min(int(self.cfg.NEGATIVE_RATIO), len(neg_candidates))
                if n_neg > 0:
                    neg_items = np.random.choice(neg_candidates, size=n_neg, replace=False)
                    for neg_item in neg_items:
                        # 找到neg_item的类别（从user_hist中找）
                        neg_cates = [int(h[2]) for h in user_hist if int(h[1]) == neg_item]
                        neg_cate = neg_cates[0] if neg_cates else 0

                        neg_cate_feat = cate_stats.get(neg_cate,
                                                       {'cate_view': 0, 'cate_cart': 0, 'cate_buy': 0, 'cate_ctr': 0,
                                                        'cate_cvr': 0})

                        neg_sample = [
                            0,  # label
                            recent_stats['n_view'], recent_stats['n_fav'], recent_stats['n_cart'],
                            recent_stats['n_buy'],
                            recent_stats['len'], recent_stats['unique_items'], recent_stats['unique_cates'],
                            user_global['user_total_buy'], user_global['user_total_view'], user_global['user_buy_rate'],
                            neg_cate_feat['cate_view'], neg_cate_feat['cate_cart'], neg_cate_feat['cate_buy'],
                            neg_cate_feat['cate_ctr'], neg_cate_feat['cate_cvr'],
                            hour, date_type
                        ]
                        target_list.append(neg_sample)

        # 创建DataFrame
        columns = ['label',
                   'recent_view', 'recent_fav', 'recent_cart', 'recent_buy', 'recent_len', 'recent_unique_items',
                   'recent_unique_cates',
                   'user_total_buy', 'user_total_view', 'user_buy_rate',
                   'cate_view', 'cate_cart', 'cate_buy', 'cate_ctr', 'cate_cvr',
                   'hour', 'date_type']

        train_df = pd.DataFrame(train_samples, columns=columns) if train_samples else pd.DataFrame(columns=columns)
        val_df = pd.DataFrame(val_samples, columns=columns) if val_samples else pd.DataFrame(columns=columns)
        test_df = pd.DataFrame(test_samples, columns=columns) if test_samples else pd.DataFrame(columns=columns)

        print(f"      Train: {len(train_df):,} (正{(train_df['label'] == 1).sum():,})")
        print(f"      Val:   {len(val_df):,} (正{(val_df['label'] == 1).sum():,})")
        print(f"      Test:  {len(test_df):,} (正{(test_df['label'] == 1).sum():,})")

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            return train_df, val_df, test_df

        return train_df, val_df, test_df

    def _build_user_index(self, df):
        user_histories = defaultdict(list)
        df_sorted = df.sort_values(['user_id', 'datetime'])
        for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="构建索引"):
            user_histories[int(row['user_id'])].append([
                row['datetime'],
                int(row['item_id']),
                int(row['item_category']),
                str(row['behavior_type'])
            ])
        for uid in user_histories:
            user_histories[uid] = np.array(user_histories[uid], dtype=object)
        return user_histories

    def _get_user_history_before(self, user_histories, user_id, cutoff_time):
        if user_id not in user_histories:
            return []
        hist = user_histories[user_id]
        times = np.array([h[0] for h in hist])
        mask = times < cutoff_time
        if not np.any(mask):
            return []
        return hist[mask]

    def _fast_compute_stats(self, user_hist):
        if len(user_hist) == 0:
            return {'n_view': 0, 'n_fav': 0, 'n_cart': 0, 'n_buy': 0,
                    'len': 0, 'unique_items': 0, 'unique_cates': 0}
        behaviors = user_hist[:, 3]
        return {
            'n_view': int(np.sum(behaviors == '1')),
            'n_fav': int(np.sum(behaviors == '2')),
            'n_cart': int(np.sum(behaviors == '3')),
            'n_buy': int(np.sum(behaviors == '4')),
            'len': int(len(user_hist)),
            'unique_items': len(set(int(h[1]) for h in user_hist)),
            'unique_cates': len(set(int(h[2]) for h in user_hist))
        }

    def _get_date_type(self, date_str):
        promo = ['2014-12-11', '2014-12-12', '2014-12-13']
        weekend = ['2014-11-22', '2014-11-23', '2014-11-29', '2014-11-30',
                   '2014-12-06', '2014-12-07', '2014-12-13', '2014-12-14']
        if date_str in promo:
            return 2
        elif date_str in weekend:
            return 1
        else:
            return 0


class StatOnlyDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        if len(df) == 0:
            return

        # 所有特征都是数值型，无需embedding
        feature_cols = [c for c in df.columns if c != 'label']
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if len(self.df) == 0:
            return torch.tensor([]), torch.tensor(0.0)
        return self.features[idx], self.labels[idx]


class StatOnlyMLP(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in cfg.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.DROPOUT_RATE)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[模型] 纯统计特征MLP，输入维度: {input_dim}，参数量: {total_params / 1e6:.3f}M")

    def forward(self, x):
        x = self.mlp(x)
        return torch.sigmoid(self.output(x)).squeeze(1)


def train_model(model, train_loader, val_loader, cfg):
    model.to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    criterion = nn.BCELoss()

    best_val_auc = 0
    best_model_state = None
    patience_counter = 0

    print(f"\n开始训练（纯统计特征，早停耐心{cfg.EARLY_STOPPING_PATIENCE}）...")

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
            features = features.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_auc = roc_auc_score(train_labels, train_preds)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} Val", leave=False):
                features = features.to(cfg.DEVICE)
                val_preds.extend(model(features).cpu().numpy())
                val_labels.extend(labels.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(val_auc)

        gap = train_auc - val_auc
        print(f"Epoch {epoch + 1:2d}: Loss={np.mean(train_losses):.4f} | "
              f"Train={train_auc:.4f} | Val={val_auc:.4f} | Gap={gap:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"      ✓ 新的最佳 (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"\n[!] 早停，最佳Val AUC: {best_val_auc:.4f}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(cfg.DEVICE)
    return model, best_val_auc


def evaluate_model(model, test_loader, cfg):
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing"):
            features = features.to(cfg.DEVICE)
            test_preds.extend(model(features).cpu().numpy())
            test_labels.extend(labels.numpy())

    test_auc = roc_auc_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, (np.array(test_preds) > 0.5).astype(int))

    print(f"\n{'=' * 60}")
    print(f"纯统计特征结果: Test AUC={test_auc:.4f} | Test F1={test_f1:.4f}")
    print(f"对比XGBoost(0.8201): 这是冷启动 vs 非冷启动的合理差距")
    print(f"对比上版(0.6061): 提升: {test_auc - 0.6061:+.4f} (验证ID特征是否冗余)")
    print(f"{'=' * 60}")
    return test_auc


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    print("=" * 60)
    print("商品冷启动MLP - 纯统计特征版 (无ID Embedding)")
    print(f"设备: {cfg.DEVICE}")
    print("=" * 60)

    builder = StatOnlyBuilder(cfg)
    train_df, val_df, test_df = builder.build_samples()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("错误：数据集为空")
        return

    # 获取输入维度
    input_dim = train_df.shape[1] - 1  # 去掉label列

    train_loader = DataLoader(StatOnlyDataset(train_df), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StatOnlyDataset(val_df), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StatOnlyDataset(test_df), batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = StatOnlyMLP(input_dim, cfg)
    model, best_val_auc = train_model(model, train_loader, val_loader, cfg)
    test_auc = evaluate_model(model, test_loader, cfg)

    results = {
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'input_dim': input_dim
    }
    with open(os.path.join(cfg.SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'model.pth'))
    print(f"\n结果已保存至: {cfg.SAVE_DIR}")


if __name__ == "__main__":
    main()