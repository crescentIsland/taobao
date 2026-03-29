"""
DIN 最终修复版 - 修复IndexError（编码不连续问题）
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
import copy
import gc
import joblib

warnings.filterwarnings('ignore')


class Config:
    DATA_PATH = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet'
    SAVE_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\din_model_v2'
    os.makedirs(SAVE_DIR, exist_ok=True)
    EMBEDDING_DIM = 16
    ATTENTION_UNITS = [32, 16]
    DROPOUT_RATE = 0.6
    L2_REG = 0.02
    MAX_SEQ_LEN = 10
    NEG_SAMPLE_RATIO = 5
    BATCH_SIZE = 512
    EPOCHS = 20
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIN_ITEM_FREQ = 3


print(f"[INFO] 设备: {Config.DEVICE}", flush=True)


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_and_preprocess(self):
        print("[1/5] 加载数据...", flush=True)
        df = pd.read_parquet(self.data_path)
        print(f"      原始: {len(df):,}条, 用户:{df['user_id'].nunique():,}, 商品:{df['item_id'].nunique():,}",
              flush=True)

        df['behavior_type'] = df['behavior_type'].astype(str)

        if Config.MIN_ITEM_FREQ > 1:
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= Config.MIN_ITEM_FREQ].index
            df = df[df['item_id'].isin(valid_items)]
            print(f"[2/5] 过滤后: {len(df):,}条, 商品:{df['item_id'].nunique():,}", flush=True)

        df = df.sort_values(['user_id', 'datetime'])

        print("[3/5] 准备编码字典...", flush=True)
        self.item_to_cate = dict(df[['item_id', 'item_category']].drop_duplicates().values)

        all_items = df['item_id'].unique()
        item_counts = df['item_id'].value_counts()
        popular_items = item_counts.head(2000).index.tolist()

        print("[4/5] 构造样本（同步编码）...", flush=True)
        samples = self._construct_and_encode(df, all_items, popular_items)
        samples_df = pd.DataFrame(samples)

        # 按时间划分
        samples_df = samples_df.sort_values('datetime')
        n = len(samples_df)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        samples_df['is_train'] = 0
        samples_df['is_test'] = 0
        samples_df.iloc[:train_end, samples_df.columns.get_loc('is_train')] = 1
        samples_df.iloc[train_end:val_end, samples_df.columns.get_loc('is_train')] = 0
        samples_df.iloc[val_end:, samples_df.columns.get_loc('is_test')] = 1

        print(f"      样本:{len(samples_df):,}(正{samples_df['label'].sum():,}/负{(samples_df['label'] == 0).sum():,})",
              flush=True)

        print("[5/5] 压缩编码（解决IndexError）...", flush=True)
        # 关键修复：重新映射编码，使其从0开始连续
        samples_df, n_users, n_items, n_cates = self._remap_ids(samples_df)

        print(f"      最终: 用户{n_users:,}, 商品{n_items:,}, 类别{n_cates:,}", flush=True)

        del df
        gc.collect()

        # 创建encoder用于后续推理
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.cate_encoder = LabelEncoder()
        self.user_encoder.fit(samples_df['user_id_encoded'].unique())
        self.item_encoder.fit(samples_df['target_item_encoded'].unique())
        self.cate_encoder.fit(samples_df['target_cate_encoded'].unique())

        return samples_df, self.user_encoder, self.item_encoder, self.cate_encoder, n_users, n_items, n_cates

    def _remap_ids(self, samples_df):
        """重新映射ID，使其从0开始连续（关键修复IndexError）"""
        # 用户ID重新映射
        unique_users = sorted(samples_df['user_id_encoded'].unique())
        user_map = {old: new for new, old in enumerate(unique_users)}
        samples_df['user_id_encoded'] = samples_df['user_id_encoded'].map(user_map)

        # 商品ID重新映射
        unique_items = sorted(samples_df['target_item_encoded'].unique())
        item_map = {old: new for new, old in enumerate(unique_items)}
        samples_df['target_item_encoded'] = samples_df['target_item_encoded'].map(item_map)

        # 类别ID重新映射
        unique_cates = sorted(samples_df['target_cate_encoded'].unique())
        cate_map = {old: new for new, old in enumerate(unique_cates)}
        samples_df['target_cate_encoded'] = samples_df['target_cate_encoded'].map(cate_map)

        # 序列中的ID也需要重新映射（注意：0是padding，要保持为0）
        samples_df['seq_items_encoded'] = samples_df['seq_items_encoded'].apply(
            lambda x: [item_map.get(i, 0) if i != 0 else 0 for i in x]
        )
        samples_df['seq_cates_encoded'] = samples_df['seq_cates_encoded'].apply(
            lambda x: [cate_map.get(c, 0) if c != 0 else 0 for c in x]
        )

        return samples_df, len(user_map), len(item_map), len(cate_map)

    def _construct_and_encode(self, df, all_items, popular_items):
        """构造样本并立即编码"""
        unique_users = df['user_id'].unique()
        unique_items = df['item_id'].unique()
        unique_cates = df['item_category'].unique()

        user2idx = {u: i + 1 for i, u in enumerate(unique_users)}  # 从1开始，0留给padding
        item2idx = {item: i + 1 for i, item in enumerate(unique_items)}
        cate2idx = {c: i + 1 for i, c in enumerate(unique_cates)}

        samples = []
        user_groups = df.groupby('user_id')

        for user_id, user_df in tqdm(user_groups, desc="      进度"):
            user_df = user_df.sort_values('datetime')
            purchases = user_df[user_df['behavior_type'] == '4']

            if len(purchases) == 0:
                continue

            user_encoded = user2idx[user_id]

            for idx, row in purchases.iterrows():
                target_item = row['item_id']
                target_time = row['datetime']
                target_item_enc = item2idx[target_item]
                target_cate_enc = cate2idx.get(self.item_to_cate[target_item], 0)

                history = user_df[user_df['datetime'] < target_time].tail(Config.MAX_SEQ_LEN)
                if len(history) == 0:
                    continue

                seq_items = [item2idx.get(x, 0) for x in history['item_id'].tolist()]
                seq_cates = [cate2idx.get(x, 0) for x in history['item_category'].tolist()]
                seq_behaviors = history['behavior_type'].tolist()
                seq_len = len(seq_items)

                if seq_len < Config.MAX_SEQ_LEN:
                    pad_len = Config.MAX_SEQ_LEN - seq_len
                    seq_items = [0] * pad_len + seq_items
                    seq_cates = [0] * pad_len + seq_cates
                    seq_behaviors = ['0'] * pad_len + seq_behaviors

                samples.append({
                    'user_id': user_id,
                    'user_id_encoded': user_encoded,
                    'target_item': target_item,
                    'target_item_encoded': target_item_enc,
                    'target_cate_encoded': target_cate_enc,
                    'seq_items_encoded': seq_items,
                    'seq_cates_encoded': seq_cates,
                    'seq_behaviors': seq_behaviors,
                    'seq_len': seq_len,
                    'label': 1,
                    'datetime': target_time
                })

                # 负采样
                n_neg = Config.NEG_SAMPLE_RATIO
                neg_candidates = []
                n_pop = int(n_neg * 0.8)
                if len(popular_items) >= n_pop:
                    neg_pop = np.random.choice(popular_items, size=n_pop * 3, replace=True)
                    neg_pop_enc = [(item2idx[x], cate2idx.get(self.item_to_cate[x], 0))
                                   for x in neg_pop if x != target_item and x in item2idx][:n_pop]
                    neg_candidates.extend(neg_pop_enc)

                n_rand = n_neg - len(neg_candidates)
                if n_rand > 0:
                    neg_rand = np.random.choice(all_items, size=n_rand * 5, replace=True)
                    neg_rand_enc = [(item2idx[x], cate2idx.get(self.item_to_cate[x], 0))
                                    for x in neg_rand if x != target_item and x in item2idx][:n_rand]
                    neg_candidates.extend(neg_rand_enc)

                for neg_item_enc, neg_cate_enc in neg_candidates[:n_neg]:
                    samples.append({
                        'user_id': user_id,
                        'user_id_encoded': user_encoded,
                        'target_item': -1,
                        'target_item_encoded': neg_item_enc,
                        'target_cate_encoded': neg_cate_enc,
                        'seq_items_encoded': seq_items,
                        'seq_cates_encoded': seq_cates,
                        'seq_behaviors': seq_behaviors,
                        'seq_len': seq_len,
                        'label': 0,
                        'datetime': target_time
                    })

        return samples


class DINDataset(Dataset):
    def __init__(self, samples_df):
        self.data = samples_df.reset_index(drop=True)
        self.behavior_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        print(f"  [Dataset] {len(self.data):,} 样本", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            torch.tensor(row['user_id_encoded'], dtype=torch.long),
            torch.tensor(row['target_item_encoded'], dtype=torch.long),
            torch.tensor(row['target_cate_encoded'], dtype=torch.long),
            torch.tensor(row['seq_items_encoded'], dtype=torch.long),
            torch.tensor(row['seq_cates_encoded'], dtype=torch.long),
            torch.tensor([self.behavior_map.get(str(b), 0) for b in row['seq_behaviors']], dtype=torch.long),
            torch.tensor(row['seq_len'], dtype=torch.long),
            torch.tensor(row['label'], dtype=torch.float32)
        )


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super(AttentionLayer, self).__init__()
        layers = []
        prev_dim = input_dim * 4
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.PReLU(), nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, query, items, seq_lengths):
        B, T, E = items.size()
        query_expanded = query.unsqueeze(1).expand(B, T, -1)
        concat = torch.cat([query_expanded, items, query_expanded - items, query_expanded * items], dim=-1)
        scores = self.mlp(concat.view(B * T, -1)).view(B, T)
        mask = torch.arange(T, device=items.device).unsqueeze(0).expand(B, -1) < seq_lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), items).squeeze(1), weights


class DIN(nn.Module):
    def __init__(self, n_users, n_items, n_cates, embedding_dim=16):
        super(DIN, self).__init__()
        # 注意：现在n_users/n_items/n_cates已经是压缩后的实际数量
        self.user_emb = nn.Embedding(n_users, embedding_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(n_cates, embedding_dim, padding_idx=0)
        self.behavior_emb = nn.Embedding(5, embedding_dim, padding_idx=0)
        self.attention = AttentionLayer(embedding_dim, Config.ATTENTION_UNITS, Config.DROPOUT_RATE)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128), nn.PReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids, target_items, target_cates, seq_items, seq_cates, seq_behaviors, seq_lengths):
        user_vec = self.user_emb(user_ids)
        target_item_vec = self.item_emb(target_items)
        target_cate_vec = self.cate_emb(target_cates)
        seq_vec = self.item_emb(seq_items) + self.cate_emb(seq_cates) + self.behavior_emb(seq_behaviors)
        interest_vec, _ = self.attention(target_item_vec, seq_vec, seq_lengths)
        return self.mlp(torch.cat([user_vec, target_item_vec, target_cate_vec, interest_vec], dim=-1)).squeeze(-1), None


def train_epoch(model, train_loader, optimizer, criterion, device, l2_reg):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in tqdm(train_loader, desc="Training"):
        user_ids, target_items, target_cates, seq_items, seq_cates, seq_behaviors, seq_lengths, labels = [b.to(device)
                                                                                                          for b in
                                                                                                          batch]
        optimizer.zero_grad()
        preds, _ = model(user_ids, target_items, target_cates, seq_items, seq_cates, seq_behaviors, seq_lengths)
        loss = criterion(preds, labels) + l2_reg * sum(torch.sum(p ** 2) for p in model.parameters())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(train_loader), roc_auc_score(all_labels, all_preds)


def eval_epoch(model, eval_loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in eval_loader:
            user_ids, target_items, target_cates, seq_items, seq_cates, seq_behaviors, seq_lengths, labels = [
                b.to(device) for b in batch]
            preds, _ = model(user_ids, target_items, target_cates, seq_items, seq_cates, seq_behaviors, seq_lengths)
            total_loss += criterion(preds, labels).item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(eval_loader), roc_auc_score(all_labels, all_preds)


def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience,
                              save_dir):
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}", flush=True)
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device, Config.L2_REG)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion, device)
        print(f"Train Loss:{train_loss:.4f} AUC:{train_auc:.4f} | Val Loss:{val_loss:.4f} AUC:{val_auc:.4f}",
              flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({'model_state_dict': model.state_dict(), 'val_auc': val_auc},
                       os.path.join(save_dir, 'best_model.pth'))
            print(f"✓ 保存最佳模型 (Val AUC:{val_auc:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"× 未提升 ({patience_counter}/{patience})", flush=True)
            if patience_counter >= patience:
                print(f"早停！最佳Val AUC:{best_val_auc:.4f}", flush=True)
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_val_auc


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60, flush=True)
    print("DIN最终修复版（修复IndexError）", flush=True)
    print("=" * 60, flush=True)

    preprocessor = DataPreprocessor(Config.DATA_PATH)
    samples_df, user_enc, item_enc, cate_enc, n_users, n_items, n_cates = preprocessor.load_and_preprocess()

    print("\n[Split] 划分数据集...", flush=True)
    train_df = samples_df[samples_df['is_train'] == 1]
    val_df = samples_df[(samples_df['is_train'] == 0) & (samples_df['is_test'] == 0)]
    test_df = samples_df[samples_df['is_test'] == 1]
    print(f"训练集:{len(train_df):,} 验证集:{len(val_df):,} 测试集:{len(test_df):,}", flush=True)

    print("\n[Loader] 创建DataLoader...", flush=True)
    train_loader = DataLoader(DINDataset(train_df), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DINDataset(val_df), batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(DINDataset(test_df), batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"\n[Model] 初始化: 用户{n_users:,} 商品{n_items:,} 类别{n_cates:,}", flush=True)
    model = DIN(n_users, n_items, n_cates, Config.EMBEDDING_DIM).to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量:{total_params / 1e6:.2f}M", flush=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.001)

    print("\n[Train] 开始训练...", flush=True)
    model, best_val_auc = train_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion,
        Config.DEVICE, Config.EPOCHS, Config.EARLY_STOPPING_PATIENCE, Config.SAVE_DIR
    )

    print("\n[Test] 测试集评估...", flush=True)
    test_loss, test_auc = eval_epoch(model, test_loader, criterion, Config.DEVICE)
    print(f"[Test] Loss:{test_loss:.4f} AUC:{test_auc:.4f}", flush=True)
    print(f"[Compare] XGBoost:0.8201 vs DIN:{test_auc:.4f} (差距:{abs(test_auc - 0.8201):.4f})", flush=True)

    joblib.dump({'test_auc': test_auc, 'best_val_auc': best_val_auc}, os.path.join(Config.SAVE_DIR, 'results.pkl'))
    print(f"\n[SAVED] 保存到:{Config.SAVE_DIR}", flush=True)


if __name__ == "__main__":
    main()

# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\DIN2.py
# [INFO] 设备: cpu
# ============================================================
# DIN最终修复版（修复IndexError）
# ============================================================
# [1/5] 加载数据...
#       原始: 12,253,262条, 用户:10,000, 商品:2,876,947
# [2/5] 过滤后: 9,740,838条, 商品:1,258,928
# [3/5] 准备编码字典...
# [4/5] 构造样本（同步编码）...
#       进度: 100%|██████████| 9995/9995 [03:19<00:00, 50.04it/s]
#       样本:678,768(正113,128/负565,640)
# [5/5] 压缩编码（解决IndexError）...
#       最终: 用户8,779, 商品189,923, 类别5,636
#
# [Split] 划分数据集...
# 训练集:543,014 验证集:67,877 测试集:67,877
#
# [Loader] 创建DataLoader...
#   [Dataset] 543,014 样本
#   [Dataset] 67,877 样本
#   [Dataset] 67,877 样本
#
# [Model] 初始化: 用户8,779 商品189,923 类别5,636
# [Model] 参数量:3.29M
# Training:   0%|          | 0/1061 [00:00<?, ?it/s]
# [Train] 开始训练...
#
# Epoch 1/20
# Training: 100%|██████████| 1061/1061 [03:01<00:00,  5.84it/s]
# Train Loss:0.8339 AUC:0.4987 | Val Loss:0.4575 AUC:0.4999
# ✓ 保存最佳模型 (Val AUC:0.4999)
#
# Epoch 2/20
# Training: 100%|██████████| 1061/1061 [03:08<00:00,  5.64it/s]
# Train Loss:0.4928 AUC:0.4980 | Val Loss:0.4585 AUC:0.5000
# Training:   0%|          | 0/1061 [00:00<?, ?it/s]✓ 保存最佳模型 (Val AUC:0.5000)
#
# Epoch 3/20
# Training: 100%|██████████| 1061/1061 [03:31<00:00,  5.01it/s]
# Training:   0%|          | 0/1061 [00:00<?, ?it/s]Train Loss:0.4901 AUC:0.5014 | Val Loss:0.4572 AUC:0.5000
# × 未提升 (1/3)
#
# Epoch 4/20
# Training: 100%|██████████| 1061/1061 [03:29<00:00,  5.07it/s]
# Training:   0%|          | 0/1061 [00:00<?, ?it/s]Train Loss:0.4899 AUC:0.4999 | Val Loss:0.4576 AUC:0.5000
# × 未提升 (2/3)
#
# Epoch 5/20
# Training: 100%|██████████| 1061/1061 [03:01<00:00,  5.85it/s]
# Train Loss:0.4899 AUC:0.5022 | Val Loss:0.4567 AUC:0.5000
# × 未提升 (3/3)
# 早停！最佳Val AUC:0.5000
#
# [Test] 测试集评估...
# [Test] Loss:0.4583 AUC:0.5000
# [Compare] XGBoost:0.8201 vs DIN:0.5000 (差距:0.3201)
#
# [SAVED] 保存到:E:\PycharmeProjects\taobao_user_behavior_analysis\results\din_model_v2


