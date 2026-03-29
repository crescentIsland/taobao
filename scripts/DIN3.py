"""
极简MLP最终版 - 基于已验证的0.76 AUC版本
去掉有bug的Attention，专注统计特征+历史行为
预期AUC: 0.78-0.82
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
import joblib

warnings.filterwarnings('ignore')


class Config:
    DATA_PATH = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet'
    SAVE_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\mlp_final'
    os.makedirs(SAVE_DIR, exist_ok=True)
    EMBEDDING_DIM = 32
    BATCH_SIZE = 256
    EPOCHS = 20
    LR = 0.001
    DROPOUT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIN_ITEM_FREQ = 3


print(f"[INFO] 设备: {Config.DEVICE}")


class DataPreprocessor:
    def load_and_preprocess(self):
        print("[1/3] 加载数据...")
        df = pd.read_parquet(Config.DATA_PATH)
        print(f"      原始: {len(df):,}条")

        if Config.MIN_ITEM_FREQ > 1:
            valid_items = df['item_id'].value_counts()[lambda x: x >= Config.MIN_ITEM_FREQ].index
            df = df[df['item_id'].isin(valid_items)]

        df = df.sort_values(['user_id', 'datetime'])
        item_to_cate = dict(df[['item_id', 'item_category']].drop_duplicates().values)

        print("[2/3] 构造样本...")
        samples = []
        for user_id, user_df in tqdm(df.groupby('user_id'), desc="构造"):
            user_df = user_df.sort_values('datetime')
            purchases = user_df[user_df['behavior_type'] == '4']

            if len(purchases) == 0:
                continue

            for _, row in purchases.iterrows():
                target_item = row['item_id']
                target_time = row['datetime']
                history = user_df[user_df['datetime'] < target_time].tail(15)

                if len(history) == 0:
                    continue

                # 统计特征
                n_view = len(history[history['behavior_type'] == '1'])
                n_fav = len(history[history['behavior_type'] == '2'])
                n_cart = len(history[history['behavior_type'] == '3'])
                n_buy = len(history[history['behavior_type'] == '4'])

                # 最近3个商品
                recent_items = history['item_id'].tolist()[-3:]
                recent_cates = history['item_category'].tolist()[-3:]
                while len(recent_items) < 3:
                    recent_items = [0] + recent_items
                    recent_cates = [0] + recent_cates

                # 正样本
                samples.append({
                    'user_id': user_id, 'target_item': target_item,
                    'target_cate': item_to_cate.get(target_item, 0),
                    'r_i1': recent_items[-1], 'r_i2': recent_items[-2], 'r_i3': recent_items[-3],
                    'r_c1': recent_cates[-1], 'r_c2': recent_cates[-2], 'r_c3': recent_cates[-3],
                    'n_view': n_view, 'n_fav': n_fav, 'n_cart': n_cart, 'n_buy': n_buy,
                    'label': 1
                })

                # 负样本
                other = user_df[user_df['behavior_type'] != '4']['item_id'].unique()
                if len(other) > 0:
                    neg = random.choice(other)
                    samples.append({
                        'user_id': user_id, 'target_item': neg,
                        'target_cate': item_to_cate.get(neg, 0),
                        'r_i1': recent_items[-1], 'r_i2': recent_items[-2], 'r_i3': recent_items[-3],
                        'r_c1': recent_cates[-1], 'r_c2': recent_cates[-2], 'r_c3': recent_cates[-3],
                        'n_view': n_view, 'n_fav': n_fav, 'n_cart': n_cart, 'n_buy': n_buy,
                        'label': 0
                    })

        samples_df = pd.DataFrame(samples)
        print(
            f"[3/3] 样本: {len(samples_df)} (正{(samples_df['label'] == 1).sum()}/负{(samples_df['label'] == 0).sum()})")

        # 编码
        cols = ['user_id', 'target_item', 'target_cate', 'r_i1', 'r_i2', 'r_i3', 'r_c1', 'r_c2', 'r_c3']
        encoders = {col: LabelEncoder().fit(samples_df[col].astype(str)) for col in cols}
        for col in cols:
            samples_df[col] = encoders[col].transform(samples_df[col].astype(str))

        # 划分
        n = len(samples_df)
        return (samples_df.iloc[:int(n * 0.8)], samples_df.iloc[int(n * 0.8):int(n * 0.9)],
                samples_df.iloc[int(n * 0.9):], encoders)


class MLP(nn.Module):
    def __init__(self, n_users, n_items, n_cates):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, 32)
        self.item_emb = nn.Embedding(n_items, 32)
        self.cate_emb = nn.Embedding(n_cates, 32)
        self.hist_emb = nn.Embedding(n_items, 32)

        # 输入: 32*9 + 4 = 292
        self.mlp = nn.Sequential(
            nn.Linear(292, 512), nn.ReLU(), nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(Config.DROPOUT),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, u, i, c, i1, i2, i3, c1, c2, c3, nv, nf, nc, nb):
        x = torch.cat([
            self.user_emb(u), self.item_emb(i), self.cate_emb(c),
            self.hist_emb(i1), self.hist_emb(i2), self.hist_emb(i3),
            self.cate_emb(c1), self.cate_emb(c2), self.cate_emb(c3),
            torch.stack([nv, nf, nc, nb], dim=1)
        ], dim=1)
        return self.mlp(x).squeeze(-1)


class Dataset(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return (torch.tensor(r['user_id'], dtype=torch.long),
                torch.tensor(r['target_item'], dtype=torch.long),
                torch.tensor(r['target_cate'], dtype=torch.long),
                torch.tensor(r['r_i1'], dtype=torch.long),
                torch.tensor(r['r_i2'], dtype=torch.long),
                torch.tensor(r['r_i3'], dtype=torch.long),
                torch.tensor(r['r_c1'], dtype=torch.long),
                torch.tensor(r['r_c2'], dtype=torch.long),
                torch.tensor(r['r_c3'], dtype=torch.long),
                torch.tensor(r['n_view'], dtype=torch.float),
                torch.tensor(r['n_fav'], dtype=torch.float),
                torch.tensor(r['n_cart'], dtype=torch.float),
                torch.tensor(r['n_buy'], dtype=torch.float),
                torch.tensor(r['label'], dtype=torch.float))


def main():
    random.seed(42);
    np.random.seed(42);
    torch.manual_seed(42)

    print("=" * 60)
    print("极简MLP最终版（去掉Attention，专注统计特征）")
    print("=" * 60)

    train_df, val_df, test_df, encoders = DataPreprocessor().load_and_preprocess()
    n_users = len(encoders['user_id'].classes_)
    n_items = len(encoders['target_item'].classes_)
    n_cates = len(encoders['target_cate'].classes_)

    print(f"\n用户{n_users}, 商品{n_items}, 类别{n_cates}")
    print(f"训练{len(train_df)}/验证{len(val_df)}/测试{len(test_df)}")

    train_loader = DataLoader(Dataset(train_df), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Dataset(val_df), batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(Dataset(test_df), batch_size=Config.BATCH_SIZE)

    model = MLP(n_users, n_items, n_cates).to(Config.DEVICE)
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_auc = 0
    best_model = None

    for epoch in range(Config.EPOCHS):
        model.train()
        preds, labels = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch = [b.to(Config.DEVICE) for b in batch]
            u, i, c, i1, i2, i3, c1, c2, c3, nv, nf, nc, nb, y = batch

            optimizer.zero_grad()
            p = model(u, i, c, i1, i2, i3, c1, c2, c3, nv, nf, nc, nb)
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()

            preds.extend(p.detach().cpu().numpy())
            labels.extend(y.cpu().numpy())

        train_auc = roc_auc_score(labels, preds)

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) for b in batch]
                u, i, c, i1, i2, i3, c1, c2, c3, nv, nf, nc, nb, y = batch
                val_preds.extend(model(u, i, c, i1, i2, i3, c1, c2, c3, nv, nf, nc, nb).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(val_auc)

        print(f"Epoch {epoch + 1}: Train AUC={train_auc:.4f} | Val AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()

    # 测试
    if best_model:
        model.load_state_dict(best_model)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(Config.DEVICE) for b in batch]
            u, i, c, i1, i2, i3, c1, c2, c3, nv, nf, nc, nb, y = batch
            test_preds.extend(model(u, i, c, i1, i2, i3, c1, c2, c3, nv, nf, nc, nb).cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_auc = roc_auc_score(test_labels, test_preds)
    print(f"\n[结果] Test AUC: {test_auc:.4f}")
    print(f"       XGBoost: 0.8201")
    print(f"       差距: {abs(test_auc - 0.8201):.4f}")

    joblib.dump({'auc': test_auc}, os.path.join(Config.SAVE_DIR, 'results.pkl'))


if __name__ == "__main__":
    main()


# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\DIN3.py
# [INFO] 设备: cpu
# ============================================================
# 极简MLP最终版（去掉Attention，专注统计特征）
# ============================================================
# [1/3] 加载数据...
#       原始: 12,253,262条
# [2/3] 构造样本...
# 构造: 100%|██████████| 9995/9995 [03:03<00:00, 54.50it/s]
# [3/3] 样本: 226256 (正113128/负113128)
#
# 用户8779, 商品171338, 类别5361
# 训练181004/验证22626/测试22626
#
# 参数量: 11.73M
# Epoch 1: 100%|██████████| 708/708 [01:19<00:00,  8.88it/s]
# Epoch 1: Train AUC=0.5940 | Val AUC=0.6408
# Epoch 2: 100%|██████████| 708/708 [01:16<00:00,  9.26it/s]
# Epoch 2: Train AUC=0.6694 | Val AUC=0.6650
# Epoch 3: 100%|██████████| 708/708 [01:15<00:00,  9.39it/s]
# Epoch 3: Train AUC=0.7314 | Val AUC=0.6684
# Epoch 4: 100%|██████████| 708/708 [01:18<00:00,  9.02it/s]
# Epoch 4: Train AUC=0.7982 | Val AUC=0.6536
# Epoch 5: 100%|██████████| 708/708 [01:18<00:00,  9.04it/s]
# Epoch 5: Train AUC=0.8587 | Val AUC=0.6323
# Epoch 6: 100%|██████████| 708/708 [01:18<00:00,  8.97it/s]
# Epoch 6: Train AUC=0.9046 | Val AUC=0.6265
# Epoch 7: 100%|██████████| 708/708 [01:18<00:00,  9.02it/s]
# Epoch 7: Train AUC=0.9369 | Val AUC=0.6256
# Epoch 8: 100%|██████████| 708/708 [01:16<00:00,  9.21it/s]
# Epoch 8: Train AUC=0.9673 | Val AUC=0.6266
# Epoch 9: 100%|██████████| 708/708 [01:14<00:00,  9.52it/s]
# Epoch 9: Train AUC=0.9778 | Val AUC=0.6216
# Epoch 10: 100%|██████████| 708/708 [01:14<00:00,  9.47it/s]
# Epoch 11:   0%|          | 0/708 [00:00<?, ?it/s]Epoch 10: Train AUC=0.9834 | Val AUC=0.6241
# Epoch 11: 100%|██████████| 708/708 [01:14<00:00,  9.49it/s]
# Epoch 12:   0%|          | 0/708 [00:00<?, ?it/s]Epoch 11: Train AUC=0.9870 | Val AUC=0.6169
# Epoch 12: 100%|██████████| 708/708 [01:18<00:00,  9.02it/s]
# Epoch 12: Train AUC=0.9914 | Val AUC=0.6182
# Epoch 13: 100%|██████████| 708/708 [01:18<00:00,  9.02it/s]
# Epoch 14:   0%|          | 0/708 [00:00<?, ?it/s]Epoch 13: Train AUC=0.9932 | Val AUC=0.6193
# Epoch 14: 100%|██████████| 708/708 [01:18<00:00,  8.99it/s]
# Epoch 15:   0%|          | 0/708 [00:00<?, ?it/s]Epoch 14: Train AUC=0.9942 | Val AUC=0.6165
# Epoch 15: 100%|██████████| 708/708 [01:18<00:00,  8.98it/s]
# Epoch 16:   0%|          | 0/708 [00:00<?, ?it/s]Epoch 15: Train AUC=0.9949 | Val AUC=0.6159
# Epoch 16: 100%|██████████| 708/708 [01:15<00:00,  9.42it/s]
# Epoch 16: Train AUC=0.9960 | Val AUC=0.6187
# Epoch 17: 100%|██████████| 708/708 [01:14<00:00,  9.47it/s]
# Epoch 17: Train AUC=0.9964 | Val AUC=0.6175
# Epoch 18: 100%|██████████| 708/708 [01:14<00:00,  9.45it/s]
# Epoch 19:   0%|          | 0/708 [00:00<?, ?it/s]Epoch 18: Train AUC=0.9967 | Val AUC=0.6176
# Epoch 19: 100%|██████████| 708/708 [01:14<00:00,  9.46it/s]
# Epoch 19: Train AUC=0.9970 | Val AUC=0.6186
# Epoch 20: 100%|██████████| 708/708 [01:15<00:00,  9.37it/s]
# Epoch 20: Train AUC=0.9974 | Val AUC=0.6165
#
# [结果] Test AUC: 0.6119
#        XGBoost: 0.8201
#        差距: 0.2082
#
# Process finished with exit code 0
