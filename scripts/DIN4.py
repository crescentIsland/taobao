"""
商品冷启动MLP - 方案A (修复类型不匹配)
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
    SAVE_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\mlp_item_coldstart_v2'
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_START = '2014-11-19'
    TEST_END = '2014-12-18'

    TRAIN_ITEM_RATIO = 0.8
    VAL_ITEM_RATIO = 0.1

    EMBEDDING_DIM = 16
    HIDDEN_DIMS = [128, 64, 32]
    DROPOUT_RATE = 0.5
    USE_DATE_TYPE = True

    BATCH_SIZE = 512
    EPOCHS = 50
    LR = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5
    LR_PATIENCE = 3

    NEGATIVE_RATIO = 1.0

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ItemColdStartBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_samples(self):
        print(f"[INFO] 加载数据: {self.cfg.DATA_PATH}")
        df = pd.read_parquet(self.cfg.DATA_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')

        print(f"      原始数据: {len(df):,}条, 时间范围: {df['date'].min()} ~ {df['date'].max()}")

        # 步骤1：商品划分（基于有购买记录的商品）
        print("[1/4] 商品冷启动划分...")

        # 先筛选目标时间范围内的购买行为用于划分
        target_df = df[(df['date'] >= self.cfg.TRAIN_START) & (df['date'] <= self.cfg.TEST_END)]

        # 关键修复：确保转换为Python int，避免numpy类型不匹配
        purchased_items = target_df[target_df['behavior_type'] == '4']['item_id'].unique()
        purchased_items = [int(x) for x in purchased_items]  # 强制转为Python int

        print(f"      目标时间段内有购买的商品: {len(purchased_items):,}")

        np.random.seed(self.cfg.SEED)
        np.random.shuffle(purchased_items)

        n_items = len(purchased_items)
        n_train = int(n_items * self.cfg.TRAIN_ITEM_RATIO)
        n_val = int(n_items * self.cfg.VAL_ITEM_RATIO)

        train_items = set(purchased_items[:n_train])
        val_items = set(purchased_items[n_train:n_train + n_val])
        test_items = set(purchased_items[n_train + n_val:])

        print(f"      训练商品: {len(train_items):,} | 验证商品: {len(val_items):,} | 测试商品: {len(test_items):,}")

        # 步骤2：构建用户历史索引（使用完整数据df，不筛选时间）
        print("[2/4] 构建用户行为索引（使用完整历史）...")
        user_histories = self._build_user_index(df)

        # 步骤3：构造样本
        print("[3/4] 构造样本...")
        item_to_cate = dict(df[['item_id', 'item_category']].drop_duplicates().values)

        # 确保item_to_cate的key也是Python int
        item_to_cate = {int(k): int(v) for k, v in item_to_cate.items()}

        train_samples, val_samples, test_samples = [], [], []

        # 调试计数
        skipped_no_hist = 0
        skipped_no_set = 0

        # 只遍历目标时间范围内的购买行为
        purchases = target_df[target_df['behavior_type'] == '4'].sort_values('datetime')

        for _, row in tqdm(purchases.iterrows(), total=len(purchases), desc="构造"):
            user_id = int(row['user_id'])
            target_item = int(row['item_id'])  # 确保是Python int
            target_cate = int(row['item_category'])
            cutoff_time = row['datetime']

            # 根据target_item决定集合
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
                skipped_no_set += 1
                continue

            # 获取历史行为
            user_hist = self._get_user_history_before(user_histories, user_id, cutoff_time)

            if len(user_hist) == 0:
                skipped_no_hist += 1
                continue

            stats = self._fast_compute_stats(user_hist)
            date_type = self._get_date_type(row['date'])

            # 正样本
            sample = [
                user_id, target_item, target_cate, date_type, 1,
                stats['n_view'], stats['n_fav'], stats['n_cart'],
                stats['n_buy'], stats['len']
            ]
            target_list.append(sample)

            # 负采样（从同集合且用户历史交互过的商品中选）
            hist_items = set(int(h[1]) for h in user_hist)
            neg_candidates = list((hist_items - {target_item}) & allowed_items)

            if neg_candidates:
                n_neg = min(int(self.cfg.NEGATIVE_RATIO), len(neg_candidates))
                if n_neg > 0:
                    neg_items = np.random.choice(neg_candidates, size=n_neg, replace=False)
                    for neg_item in neg_items:
                        neg_item = int(neg_item)
                        neg_cate = item_to_cate.get(neg_item, 0)
                        neg_sample = [
                            user_id, neg_item, neg_cate, date_type, 0,
                            stats['n_view'], stats['n_fav'], stats['n_cart'],
                            stats['n_buy'], stats['len']
                        ]
                        target_list.append(neg_sample)

        print(f"      跳过（无历史）: {skipped_no_hist}, 跳过（无集合）: {skipped_no_set}")

        # 步骤4：创建DataFrame
        print(f"[4/4] 合并与编码...")
        columns = ['user_id', 'target_item', 'target_cate', 'date_type', 'label',
                   'hist_n_view', 'hist_n_fav', 'hist_n_cart', 'hist_n_buy', 'hist_len']

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

    def _build_user_index(self, df):
        """构建用户历史索引（使用完整数据）"""
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
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else 'UNK'
                    )
                    df[col] = le.transform(df[col])

        return train_df, val_df, test_df, encoders


class ColdStartDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        if len(df) == 0:
            return

        self.features = {
            'user': torch.tensor(self.df['user_id'].values, dtype=torch.long),
            'item': torch.tensor(self.df['target_item'].values, dtype=torch.long),
            'cate': torch.tensor(self.df['target_cate'].values, dtype=torch.long),
            'date_type': torch.tensor(self.df['date_type'].values, dtype=torch.long),
            'stats': torch.tensor(self.df[['hist_n_view', 'hist_n_fav', 'hist_n_cart',
                                           'hist_n_buy', 'hist_len']].values, dtype=torch.float)
        }
        self.labels = torch.tensor(self.df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if len(self.df) == 0:
            return {}, torch.tensor(0.0)
        return {k: v[idx] for k, v in self.features.items()}, self.labels[idx]


class RegularizedMLP(nn.Module):
    def __init__(self, n_users, n_items, n_cates, cfg):
        super().__init__()
        self.cfg = cfg

        self.user_emb = nn.Embedding(n_users, cfg.EMBEDDING_DIM)
        self.item_emb = nn.Embedding(n_items, cfg.EMBEDDING_DIM)
        self.cate_emb = nn.Embedding(n_cates, cfg.EMBEDDING_DIM)

        if cfg.USE_DATE_TYPE:
            self.date_emb = nn.Embedding(3, 8)
            date_dim = 8
        else:
            date_dim = 0

        input_dim = cfg.EMBEDDING_DIM * 3 + 5 + date_dim

        print(f"[维度] 输入: {input_dim} (emb:{cfg.EMBEDDING_DIM}×3 + stats:5 + date:{date_dim})")

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
        print(f"[模型] 总参数量: {total_params / 1e6:.2f}M")

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

        stats = torch.log1p(batch['stats'])

        concat_list = [u, i, c, stats]

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

    print(f"\n开始训练（最大{cfg.EPOCHS}轮，早停耐心{cfg.EARLY_STOPPING_PATIENCE}）...")

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

        print(f"Epoch {epoch + 1:2d}: Loss={np.mean(train_losses):.4f} | "
              f"Train AUC={train_auc:.4f} | Val AUC={val_auc:.4f} | LR={current_lr:.6f}")

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
    print(f"测试结果: Test AUC={test_auc:.4f} | Test F1={test_f1:.4f}")
    print(f"对比XGBoost: 0.8201 | 差距: {abs(test_auc - 0.8201):.4f}")
    print(f"{'=' * 60}")
    return test_auc, test_preds


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    print("=" * 60)
    print("商品冷启动MLP - 方案A (修复类型匹配)")
    print(f"设备: {cfg.DEVICE}")
    print("=" * 60)

    builder = ItemColdStartBuilder(cfg)
    train_df, val_df, test_df, encoders = builder.build_samples()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("错误：数据集为空，无法训练")
        return

    train_loader = DataLoader(ColdStartDataset(train_df), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ColdStartDataset(val_df), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ColdStartDataset(test_df), batch_size=cfg.BATCH_SIZE, shuffle=False)

    n_users = len(encoders['user_id'].classes_)
    n_items = len(encoders['target_item'].classes_)
    n_cates = len(encoders['target_cate'].classes_)

    print(f"\n词汇表: 用户{n_users}, 商品{n_items}, 类别{n_cates}")

    model = RegularizedMLP(n_users, n_items, n_cates, cfg)
    model, best_val_auc, history = train_model(model, train_loader, val_loader, cfg)
    test_auc, _ = evaluate_model(model, test_loader, cfg)

    results = {
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'history': {k: [float(x) for x in v] for k, v in history.items()}
    }
    with open(os.path.join(cfg.SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'model.pth'))
    print(f"\n结果已保存至: {cfg.SAVE_DIR}")


if __name__ == "__main__":
    main()

# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\DIN4.py
# ============================================================
# 商品冷启动MLP - 方案A (修复类型匹配)
# 设备: cpu
# ============================================================
# [INFO] 加载数据: E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet
#       原始数据: 12,253,262条, 时间范围: 2014-11-18 ~ 2014-12-18
# [1/4] 商品冷启动划分...
#       目标时间段内有购买的商品: 90,078
#       训练商品: 72,062 | 验证商品: 9,007 | 测试商品: 9,009
# [2/4] 构建用户行为索引（使用完整历史）...
# 构建索引: 100%|██████████| 12253262/12253262 [08:21<00:00, 24420.58it/s]
# [3/4] 构造样本...
# 构造: 100%|██████████| 114286/114286 [04:13<00:00, 450.05it/s]
#       跳过（无历史）: 608, 跳过（无集合）: 0
# [4/4] 合并与编码...
#       Train: 179,979 (正90,854)
#       Val:   21,166 (正11,571)
#       Test:  20,476 (正11,253)
#
# 词汇表: 用户8582, 商品89893, 类别4375
# [维度] 输入: 61 (emb:16×3 + stats:5 + date:8)
# [模型] 总参数量: 1.66M
#
# 开始训练（最大50轮，早停耐心5）...
# Epoch 1 Train: 100%|██████████| 352/352 [00:08<00:00, 39.32it/s]
# Epoch 2 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  1: Loss=0.7405 | Train AUC=0.5020 | Val AUC=0.5392 | LR=0.001000
#       ✓ 新的最佳模型 (Val AUC: 0.5392)
# Epoch 2 Train: 100%|██████████| 352/352 [00:08<00:00, 40.26it/s]
# Epoch 3 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  2: Loss=0.6968 | Train AUC=0.5066 | Val AUC=0.5154 | LR=0.001000
# Epoch 3 Train: 100%|██████████| 352/352 [00:09<00:00, 37.59it/s]
# Epoch 4 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  3: Loss=0.6913 | Train AUC=0.5295 | Val AUC=0.5743 | LR=0.001000
#       ✓ 新的最佳模型 (Val AUC: 0.5743)
# Epoch 4 Train: 100%|██████████| 352/352 [00:09<00:00, 37.17it/s]
# Epoch 5 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  4: Loss=0.6527 | Train AUC=0.6592 | Val AUC=0.5850 | LR=0.001000
#       ✓ 新的最佳模型 (Val AUC: 0.5850)
# Epoch 5 Train: 100%|██████████| 352/352 [00:09<00:00, 36.39it/s]
# Epoch 6 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  5: Loss=0.5768 | Train AUC=0.7717 | Val AUC=0.6073 | LR=0.001000
#       ✓ 新的最佳模型 (Val AUC: 0.6073)
# Epoch 6 Train: 100%|██████████| 352/352 [00:11<00:00, 31.06it/s]
# Epoch 7 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  6: Loss=0.5036 | Train AUC=0.8296 | Val AUC=0.5944 | LR=0.001000
# Epoch 7 Train: 100%|██████████| 352/352 [00:10<00:00, 34.31it/s]
# Epoch 8 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  7: Loss=0.4487 | Train AUC=0.8610 | Val AUC=0.5822 | LR=0.001000
# Epoch 8 Train: 100%|██████████| 352/352 [00:10<00:00, 35.17it/s]
# Epoch 9 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  8: Loss=0.4112 | Train AUC=0.8827 | Val AUC=0.5753 | LR=0.001000
# Epoch 9 Train: 100%|██████████| 352/352 [00:09<00:00, 38.45it/s]
# Epoch 10 Train:   0%|          | 0/352 [00:00<?, ?it/s]Epoch  9: Loss=0.3792 | Train AUC=0.9012 | Val AUC=0.5603 | LR=0.000500
# Epoch 10 Train: 100%|██████████| 352/352 [00:09<00:00, 37.32it/s]
# Testing:   0%|          | 0/40 [00:00<?, ?it/s]Epoch 10: Loss=0.3313 | Train AUC=0.9269 | Val AUC=0.5590 | LR=0.000500
#
# [!] 早停触发！最佳Val AUC: 0.6073
# Testing: 100%|██████████| 40/40 [00:00<00:00, 84.98it/s]
#
# ============================================================
# 测试结果: Test AUC=0.6039 | Test F1=0.6744
# 对比XGBoost: 0.8201 | 差距: 0.2162
# ============================================================
#
# 结果已保存至: E:\PycharmeProjects\taobao_user_behavior_analysis\results\mlp_item_coldstart_v2
#
# Process finished with exit code 0
