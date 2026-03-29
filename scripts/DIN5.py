"""
商品冷启动MLP - 优化版 (小模型+丰富特征+强正则)
目标：提升冷启动AUC从0.60到0.65+
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import warnings
import json
from collections import defaultdict

warnings.filterwarnings('ignore')


class Config:
    DATA_PATH = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet'
    SAVE_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\mlp_item_coldstart_optimized'
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_START = '2014-11-19'
    TEST_END = '2014-12-18'

    TRAIN_ITEM_RATIO = 0.8
    VAL_ITEM_RATIO = 0.1

    # 模型瘦身：更小的维度
    EMBEDDING_DIM = 8  # 原16，减小防止过拟合
    HIDDEN_DIMS = [32, 16]  # 原[128,64,32]，大幅减小
    DROPOUT_RATE = 0.7  # 原0.5，更强dropout

    # 强正则
    WEIGHT_DECAY = 1e-3  # 原1e-4，10倍L2正则
    EARLY_STOPPING_PATIENCE = 3  # 原5，更快早停

    USE_DATE_TYPE = True

    BATCH_SIZE = 256  # 原512，小batch更稳定
    EPOCHS = 50
    LR = 0.001
    LR_PATIENCE = 2

    NEGATIVE_RATIO = 1.0

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OptimizedDataBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        # 预计算的全局统计（用于特征工程）
        self.cate_stats = {}  # 类别统计
        self.user_cate_pref = {}  # 用户-类别偏好

    def build_samples(self):
        print(f"[INFO] 加载数据: {self.cfg.DATA_PATH}")
        df = pd.read_parquet(self.cfg.DATA_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['hour'] = df['datetime'].dt.hour  # 新增：小时特征

        print(f"      原始数据: {len(df):,}条")

        # 步骤1：商品划分
        print("[1/5] 商品冷启动划分...")
        target_df = df[(df['date'] >= self.cfg.TRAIN_START) & (df['date'] <= self.cfg.TEST_END)]
        purchased_items = target_df[target_df['behavior_type'] == '4']['item_id'].unique()
        purchased_items = [int(x) for x in purchased_items]

        print(f"      目标时间段内有购买的商品: {len(purchased_items):,}")

        np.random.seed(self.cfg.SEED)
        np.random.shuffle(purchased_items)

        n_items = len(purchased_items)
        n_train = int(n_items * self.cfg.TRAIN_ITEM_RATIO)
        n_val = int(n_items * self.cfg.VAL_ITEM_RATIO)

        train_items = set(purchased_items[:n_train])
        val_items = set(purchased_items[n_train:n_train + n_val])
        test_items = set(purchased_items[n_train + n_val:])

        # 步骤2：预计算全局统计（用于特征工程）
        print("[2/5] 预计算全局统计...")
        self._precompute_stats(df, train_items)  # 只用训练集数据计算统计，防止泄露

        # 步骤3：构建索引
        print("[3/5] 构建用户行为索引...")
        user_histories = self._build_user_index(df)

        # 步骤4：构造样本（增强特征）
        print("[4/5] 构造样本（增强特征）...")
        item_to_cate = {int(k): int(v) for k, v in
                        dict(df[['item_id', 'item_category']].drop_duplicates().values).items()}

        train_samples, val_samples, test_samples = [], [], []

        purchases = target_df[target_df['behavior_type'] == '4'].sort_values('datetime')

        for _, row in tqdm(purchases.iterrows(), total=len(purchases), desc="构造"):
            user_id = int(row['user_id'])
            target_item = int(row['item_id'])
            target_cate = int(row['item_category'])
            cutoff_time = row['datetime']
            hour = int(row['hour'])

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

            # 基础统计特征
            stats = self._fast_compute_stats(user_hist)

            # 新增：用户-类别交互特征（关键！）
            user_cate_feat = self._get_user_cate_features(user_id, target_cate)

            # 新增：类别全局特征（关键！）
            cate_global = self.cate_stats.get(target_cate, {'cate_popularity': 0, 'cate_buy_rate': 0})

            # 新增：时间特征
            time_feat = {'hour': hour, 'is_night': 1 if 0 <= hour < 6 else 0}

            date_type = self._get_date_type(row['date'])

            # 组合所有特征
            sample = [
                user_id, target_item, target_cate, date_type, 1,
                # 用户行为统计
                stats['n_view'], stats['n_fav'], stats['n_cart'], stats['n_buy'], stats['len'],
                # 用户-类别偏好（交叉特征）
                user_cate_feat['user_cate_view'], user_cate_feat['user_cate_buy'],
                # 类别全局特征
                cate_global['cate_popularity'], cate_global['cate_buy_rate'],
                # 时间特征
                time_feat['hour'], time_feat['is_night']
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
                        neg_item = int(neg_item)
                        neg_cate = item_to_cate.get(neg_item, 0)

                        # 负样本同样计算特征
                        neg_user_cate_feat = self._get_user_cate_features(user_id, neg_cate)
                        neg_cate_global = self.cate_stats.get(neg_cate, {'cate_popularity': 0, 'cate_buy_rate': 0})

                        neg_sample = [
                            user_id, neg_item, neg_cate, date_type, 0,
                            stats['n_view'], stats['n_fav'], stats['n_cart'], stats['n_buy'], stats['len'],
                            neg_user_cate_feat['user_cate_view'], neg_user_cate_feat['user_cate_buy'],
                            neg_cate_global['cate_popularity'], neg_cate_global['cate_buy_rate'],
                            time_feat['hour'], time_feat['is_night']
                        ]
                        target_list.append(neg_sample)

        # 步骤5：创建DataFrame
        print(f"[5/5] 合并与编码...")
        columns = ['user_id', 'target_item', 'target_cate', 'date_type', 'label',
                   'hist_n_view', 'hist_n_fav', 'hist_n_cart', 'hist_n_buy', 'hist_len',
                   'user_cate_view', 'user_cate_buy',  # 新增
                   'cate_popularity', 'cate_buy_rate',  # 新增
                   'hour', 'is_night']  # 新增

        train_df = pd.DataFrame(train_samples, columns=columns) if train_samples else pd.DataFrame(columns=columns)
        val_df = pd.DataFrame(val_samples, columns=columns) if val_samples else pd.DataFrame(columns=columns)
        test_df = pd.DataFrame(test_samples, columns=columns) if test_samples else pd.DataFrame(columns=columns)

        print(f"      Train: {len(train_df):,} (正{(train_df['label'] == 1).sum():,})")
        print(f"      Val:   {len(val_df):,} (正{(val_df['label'] == 1).sum():,})")
        print(f"      Test:  {len(test_df):,} (正{(test_df['label'] == 1).sum():,})")

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print("      错误：某集合为空")
            return train_df, val_df, test_df, {}

        train_df, val_df, test_df, encoders = self._encode_features(train_df, val_df, test_df)
        return train_df, val_df, test_df, encoders

    def _precompute_stats(self, df, train_items):
        """预计算全局统计（仅用训练集商品，防止泄露）"""
        print("      计算类别统计...")
        train_df = df[df['item_id'].isin(train_items)]

        # 类别级别统计：每个类目被浏览/购买的次数（ popularity ）
        cate_behaviors = train_df.groupby(['item_category', 'behavior_type']).size().unstack(fill_value=0)
        for cate in cate_behaviors.index:
            views = cate_behaviors.loc[cate].get('1', 0)
            buys = cate_behaviors.loc[cate].get('4', 0)
            self.cate_stats[int(cate)] = {
                'cate_popularity': int(views + buys),  # 总交互次数
                'cate_buy_rate': float(buys / (views + buys + 1e-8))  # 购买转化率
            }

        print("      计算用户-类别偏好...")
        # 用户-类别偏好：用户对每个类目的历史交互
        user_cate = train_df.groupby(['user_id', 'item_category', 'behavior_type']).size().unstack(fill_value=0)
        for (uid, cate), row in user_cate.iterrows():
            uid, cate = int(uid), int(cate)
            if uid not in self.user_cate_pref:
                self.user_cate_pref[uid] = {}
            self.user_cate_pref[uid][cate] = {
                'user_cate_view': int(row.get('1', 0)),
                'user_cate_buy': int(row.get('4', 0))
            }

    def _get_user_cate_features(self, user_id, cate):
        """获取用户对该类别的偏好"""
        if user_id in self.user_cate_pref and cate in self.user_cate_pref[user_id]:
            return self.user_cate_pref[user_id][cate]
        return {'user_cate_view': 0, 'user_cate_buy': 0}

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
            return {'n_view': 0, 'n_fav': 0, 'n_cart': 0, 'n_buy': 0, 'len': 0}
        behaviors = user_hist[:, 3]
        return {
            'n_view': int(np.sum(behaviors == '1')),
            'n_fav': int(np.sum(behaviors == '2')),
            'n_cart': int(np.sum(behaviors == '3')),
            'n_buy': int(np.sum(behaviors == '4')),
            'len': int(len(user_hist))
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

    def _encode_features(self, train_df, val_df, test_df):
        id_cols = ['user_id', 'target_item', 'target_cate']
        encoders = {}
        for col in id_cols:
            le = LabelEncoder()
            train_values = train_df[col].astype(str).unique()
            if col == 'target_item':
                val_values = val_df[col].astype(str).unique()
                test_values = test_df[col].astype(str).unique()
                all_values = list(train_values) + list(val_values) + list(test_values) + ['UNK']
            else:
                all_values = list(train_values) + ['UNK']
            le.fit(all_values)
            encoders[col] = le
            for df in [train_df, val_df, test_df]:
                if len(df) > 0:
                    df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'UNK')
                    df[col] = le.transform(df[col])
        return train_df, val_df, test_df, encoders


class OptimizedDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        if len(df) == 0:
            return

        self.features = {
            'user': torch.tensor(self.df['user_id'].values, dtype=torch.long),
            'item': torch.tensor(self.df['target_item'].values, dtype=torch.long),
            'cate': torch.tensor(self.df['target_cate'].values, dtype=torch.long),
            'date_type': torch.tensor(self.df['date_type'].values, dtype=torch.long),
            # 所有数值特征归一化（log1p）
            'stats': torch.tensor(self.df[['hist_n_view', 'hist_n_fav', 'hist_n_cart',
                                           'hist_n_buy', 'hist_len']].values, dtype=torch.float),
            'user_cate': torch.tensor(self.df[['user_cate_view', 'user_cate_buy']].values, dtype=torch.float),
            'cate_global': torch.tensor(self.df[['cate_popularity', 'cate_buy_rate']].values, dtype=torch.float),
            'time': torch.tensor(self.df[['hour', 'is_night']].values, dtype=torch.float)
        }
        self.labels = torch.tensor(self.df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if len(self.df) == 0:
            return {}, torch.tensor(0.0)
        return {k: v[idx] for k, v in self.features.items()}, self.labels[idx]


class OptimizedMLP(nn.Module):
    def __init__(self, n_users, n_items, n_cates, cfg):
        super().__init__()
        self.cfg = cfg

        self.user_emb = nn.Embedding(n_users, cfg.EMBEDDING_DIM)
        self.item_emb = nn.Embedding(n_items, cfg.EMBEDDING_DIM)
        self.cate_emb = nn.Embedding(n_cates, cfg.EMBEDDING_DIM)

        if cfg.USE_DATE_TYPE:
            self.date_emb = nn.Embedding(3, 4)  # 减小date_emb维度
            date_dim = 4
        else:
            date_dim = 0

        # 输入维度计算
        # 3个emb(8*3=24) + 5维用户统计 + 2维用户-类别 + 2维类别全局 + 2维时间 + 4维date_type = 39
        input_dim = cfg.EMBEDDING_DIM * 3 + 5 + 2 + 2 + 2 + date_dim

        print(f"[维度] 输入: {input_dim} (emb:{cfg.EMBEDDING_DIM}×3 + feats:11 + date:{date_dim})")

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

        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[模型] 总参数量: {total_params / 1e6:.3f}M (原1.66M，应显著减小)")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, batch):
        u = self.user_emb(batch['user'])
        i = self.item_emb(batch['item'])
        c = self.cate_emb(batch['cate'])

        # 所有数值特征做log1p变换
        stats = torch.log1p(batch['stats'])
        user_cate = torch.log1p(batch['user_cate'])
        cate_global = batch['cate_global']  # buy_rate已经是比例，不需要log
        time_feat = batch['time']

        concat_list = [u, i, c, stats, user_cate, cate_global, time_feat]

        if self.cfg.USE_DATE_TYPE:
            d = self.date_emb(batch['date_type'])
            concat_list.append(d)

        x = torch.cat(concat_list, dim=1)
        x = self.mlp(x)
        return torch.sigmoid(self.output(x)).squeeze(1)


def train_model(model, train_loader, val_loader, cfg):
    model.to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=cfg.LR_PATIENCE
    )
    criterion = nn.BCELoss()

    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    history = {'train_auc': [], 'val_auc': [], 'train_loss': []}

    print(f"\n开始训练（小模型+强正则，早停耐心{cfg.EARLY_STOPPING_PATIENCE}）...")

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for batch_data, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
            batch = {k: v.to(cfg.DEVICE) for k, v in batch_data.items()}
            labels = labels.to(cfg.DEVICE)

            optimizer.zero_grad()
            preds = model(batch)
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
            for batch_data, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} Val", leave=False):
                batch = {k: v.to(cfg.DEVICE) for k, v in batch_data.items()}
                val_preds.extend(model(batch).cpu().numpy())
                val_labels.extend(labels.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_loss'].append(np.mean(train_losses))

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        gap = train_auc - val_auc
        print(f"Epoch {epoch + 1:2d}: Loss={np.mean(train_losses):.4f} | "
              f"Train={train_auc:.4f} | Val={val_auc:.4f} | Gap={gap:.4f} | LR={current_lr:.6f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"      ✓ 新的最佳模型 (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"\n[!] 早停触发！最佳Val AUC: {best_val_auc:.4f}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(cfg.DEVICE)
    return model, best_val_auc, history


def evaluate_model(model, test_loader, cfg):
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_data, labels in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(cfg.DEVICE) for k, v in batch_data.items()}
            test_preds.extend(model(batch).cpu().numpy())
            test_labels.extend(labels.numpy())

    test_auc = roc_auc_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, (np.array(test_preds) > 0.5).astype(int))

    print(f"\n{'=' * 60}")
    print(f"优化版测试结果: Test AUC={test_auc:.4f} | Test F1={test_f1:.4f}")
    print(f"对比基准: 0.6039 | 提升: {test_auc - 0.6039:+.4f}")
    print(f"{'=' * 60}")
    return test_auc, test_preds


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    print("=" * 60)
    print("商品冷启动MLP - 优化版 (小模型+丰富特征+强正则)")
    print(f"设备: {cfg.DEVICE}")
    print("=" * 60)

    builder = OptimizedDataBuilder(cfg)
    train_df, val_df, test_df, encoders = builder.build_samples()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("错误：数据集为空")
        return

    train_loader = DataLoader(OptimizedDataset(train_df), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(OptimizedDataset(val_df), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(OptimizedDataset(test_df), batch_size=cfg.BATCH_SIZE, shuffle=False)

    n_users = len(encoders['user_id'].classes_)
    n_items = len(encoders['target_item'].classes_)
    n_cates = len(encoders['target_cate'].classes_)

    print(f"\n词汇表: 用户{n_users}, 商品{n_items}, 类别{n_cates}")

    model = OptimizedMLP(n_users, n_items, n_cates, cfg)
    model, best_val_auc, history = train_model(model, train_loader, val_loader, cfg)
    test_auc, _ = evaluate_model(model, test_loader, cfg)

    results = {
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'config': {k: str(v) if not isinstance(v, (int, float, bool)) else v
                   for k, v in vars(cfg).items() if not k.startswith('__')},
        'history': {k: [float(x) for x in v] for k, v in history.items()}
    }
    with open(os.path.join(cfg.SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'model.pth'))
    print(f"\n结果已保存至: {cfg.SAVE_DIR}")


if __name__ == "__main__":
    main()

# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\DIN5.py
# ============================================================
# 商品冷启动MLP - 优化版 (小模型+丰富特征+强正则)
# 设备: cpu
# ============================================================
# [INFO] 加载数据: E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet
#       原始数据: 12,253,262条
# [1/5] 商品冷启动划分...
#       目标时间段内有购买的商品: 90,078
# [2/5] 预计算全局统计...
#       计算类别统计...
#       计算用户-类别偏好...
# [3/5] 构建用户行为索引...
# 构建索引: 100%|██████████| 12253262/12253262 [07:52<00:00, 25929.42it/s]
# [4/5] 构造样本（增强特征）...
# 构造: 100%|██████████| 114286/114286 [04:10<00:00, 457.02it/s]
# [5/5] 合并与编码...
#       Train: 179,979 (正90,854)
#       Val:   21,166 (正11,571)
#       Test:  20,476 (正11,253)
#
# 词汇表: 用户8582, 商品89893, 类别4375
# [维度] 输入: 39 (emb:8×3 + feats:11 + date:4)
# [模型] 总参数量: 0.825M (原1.66M，应显著减小)
# Epoch 1 Train:   0%|          | 0/704 [00:00<?, ?it/s]
# 开始训练（小模型+强正则，早停耐心3）...
# Epoch 1 Train: 100%|██████████| 704/704 [00:11<00:00, 61.77it/s]
# Epoch 2 Train:   0%|          | 0/704 [00:00<?, ?it/s]Epoch  1: Loss=0.7184 | Train=0.4990 | Val=0.5224 | Gap=-0.0234 | LR=0.001000
#       ✓ 新的最佳模型 (Val AUC: 0.5224)
# Epoch 2 Train: 100%|██████████| 704/704 [00:12<00:00, 56.94it/s]
# Epoch 3 Train:   0%|          | 0/704 [00:00<?, ?it/s]Epoch  2: Loss=0.6938 | Train=0.5025 | Val=0.5428 | Gap=-0.0404 | LR=0.001000
#       ✓ 新的最佳模型 (Val AUC: 0.5428)
# Epoch 3 Train: 100%|██████████| 704/704 [00:11<00:00, 62.04it/s]
# Epoch 4 Train:   0%|          | 0/704 [00:00<?, ?it/s]Epoch  3: Loss=0.6915 | Train=0.5281 | Val=0.6027 | Gap=-0.0746 | LR=0.001000
#       ✓ 新的最佳模型 (Val AUC: 0.6027)
# Epoch 4 Train: 100%|██████████| 704/704 [00:11<00:00, 62.54it/s]
# Epoch 5 Train:   0%|          | 0/704 [00:00<?, ?it/s]Epoch  4: Loss=0.6645 | Train=0.6371 | Val=0.5984 | Gap=0.0387 | LR=0.001000
# Epoch 5 Train: 100%|██████████| 704/704 [00:10<00:00, 65.44it/s]
# Epoch 6 Train:   0%|          | 0/704 [00:00<?, ?it/s]Epoch  5: Loss=0.6114 | Train=0.7361 | Val=0.6037 | Gap=0.1324 | LR=0.001000
#       ✓ 新的最佳模型 (Val AUC: 0.6037)
# Epoch 6 Train: 100%|██████████| 704/704 [00:11<00:00, 62.05it/s]
# Epoch 7 Train:   0%|          | 0/704 [00:00<?, ?it/s]Epoch  6: Loss=0.5673 | Train=0.7824 | Val=0.5953 | Gap=0.1871 | LR=0.001000
# Epoch 7 Train: 100%|██████████| 704/704 [00:10<00:00, 64.77it/s]
# Epoch 8 Train:   0%|          | 0/704 [00:00<?, ?it/s]Epoch  7: Loss=0.5317 | Train=0.8093 | Val=0.5913 | Gap=0.2180 | LR=0.001000
# Epoch 8 Train: 100%|██████████| 704/704 [00:10<00:00, 65.06it/s]
# Testing:   0%|          | 0/80 [00:00<?, ?it/s]Epoch  8: Loss=0.5047 | Train=0.8245 | Val=0.5839 | Gap=0.2405 | LR=0.000500
#
# [!] 早停触发！最佳Val AUC: 0.6037
# Testing: 100%|██████████| 80/80 [00:00<00:00, 109.10it/s]
#
# ============================================================
# 优化版测试结果: Test AUC=0.6061 | Test F1=0.6750
# 对比基准: 0.6039 | 提升: +0.0022
# ============================================================
#
# 结果已保存至: E:\PycharmeProjects\taobao_user_behavior_analysis\results\mlp_item_coldstart_optimized
#
# Process finished with exit code 0
