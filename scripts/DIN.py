"""
DIN (Deep Interest Network) 完整实现 - 修正字符型behavior_type
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ==================== 配置 ====================
class Config:
    DATA_PATH = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet'
    SAVE_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\din_model'
    os.makedirs(SAVE_DIR, exist_ok=True)

    EMBEDDING_DIM = 64
    ATTENTION_UNITS = [128, 64]
    DROPOUT_RATE = 0.3
    L2_REG = 0.001
    MAX_SEQ_LEN = 30
    NEG_SAMPLE_RATIO = 3

    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAMPLE_USER_N = None  # 测试时设为10000


print(f"使用设备: {Config.DEVICE}")


# ==================== 1. 数据预处理 ====================
class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.cate_encoder = LabelEncoder()

    def load_and_preprocess(self):
        print("加载数据...")
        df = pd.read_parquet(self.data_path)
        print(f"原始数据量: {len(df):,}条")

        # 确保behavior_type是字符串（即使已经是字符串也做一次转换）
        df['behavior_type'] = df['behavior_type'].astype(str)

        # 验证唯一值
        unique_behaviors = df['behavior_type'].unique()
        print(f"行为类型分布: {unique_behaviors}")
        print(df['behavior_type'].value_counts())

        if Config.SAMPLE_USER_N:
            unique_users = df['user_id'].unique()[:Config.SAMPLE_USER_N]
            df = df[df['user_id'].isin(unique_users)]
            print(f"采样后数据量: {len(df):,}条")

        df = df.sort_values(['user_id', 'datetime'])

        print("编码ID...")
        df['user_id_encoded'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_id_encoded'] = self.item_encoder.fit_transform(df['item_id'])
        df['cate_id_encoded'] = self.cate_encoder.fit_transform(df['item_category'])

        # 行为类型权重（键为字符串）
        self.behavior_weight = {
            '1': 1.0,  # 浏览
            '2': 2.0,  # 收藏
            '3': 3.0,  # 加购
            '4': 4.0  # 购买
        }

        return df

    def construct_sequences(self, df):
        print("构造行为序列...")
        grouped = df.groupby('user_id_encoded')

        samples = []

        for user_id, user_df in tqdm(grouped, desc="处理用户"):
            user_df = user_df.reset_index(drop=True)

            # 找到所有购买行为的索引（字符串比较'4'）
            buy_indices = user_df[user_df['behavior_type'] == '4'].index.tolist()

            for idx in buy_indices:
                if idx == 0:
                    continue

                target_item = user_df.loc[idx, 'item_id_encoded']
                target_cate = user_df.loc[idx, 'cate_id_encoded']

                start_idx = max(0, idx - Config.MAX_SEQ_LEN)
                history_df = user_df.loc[start_idx:idx - 1]

                seq_items = history_df['item_id_encoded'].tolist()
                seq_cates = history_df['cate_id_encoded'].tolist()
                seq_behaviors = history_df['behavior_type'].tolist()  # 保持字符串
                seq_len = len(seq_items)

                # 正样本
                samples.append({
                    'user_id': user_id,
                    'target_item': target_item,
                    'target_cate': target_cate,
                    'seq_items': seq_items,
                    'seq_cates': seq_cates,
                    'seq_behaviors': seq_behaviors,
                    'seq_len': seq_len,
                    'label': 1
                })

                # 负采样：从该用户的历史中选（排除当前购买的商品）
                if idx > 0:
                    neg_candidates = user_df.loc[:idx - 1]
                    # 排除购买行为本身（如果是购买后未买其他，这里简单处理）
                    neg_row = neg_candidates.sample(1).iloc[0]
                    neg_item = neg_row['item_id_encoded']
                    neg_cate = neg_row['cate_id_encoded']

                    if neg_item != target_item:
                        samples.append({
                            'user_id': user_id,
                            'target_item': neg_item,
                            'target_cate': neg_cate,
                            'seq_items': seq_items,
                            'seq_cates': seq_cates,
                            'seq_behaviors': seq_behaviors,
                            'seq_len': seq_len,
                            'label': 0
                        })

        samples_df = pd.DataFrame(samples)
        print(f"构造样本数: {len(samples_df):,}")
        print(f"正样本: {(samples_df['label'] == 1).sum():,}")
        print(f"负样本: {(samples_df['label'] == 0).sum():,}")
        return samples_df


# ==================== 2. PyTorch Dataset ====================
class DINDataset(Dataset):
    def __init__(self, samples_df):
        self.data = samples_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        user_id = torch.tensor(row['user_id'], dtype=torch.long)
        target_item = torch.tensor(row['target_item'], dtype=torch.long)
        target_cate = torch.tensor(row['target_cate'], dtype=torch.long)

        # 历史序列
        seq_items = row['seq_items'][-Config.MAX_SEQ_LEN:]
        seq_cates = row['seq_cates'][-Config.MAX_SEQ_LEN:]
        seq_behaviors = row['seq_behaviors'][-Config.MAX_SEQ_LEN:]  # 字符串列表
        seq_len = min(row['seq_len'], Config.MAX_SEQ_LEN)

        # 将字符串behavior_type转为整数（用于Embedding查询）
        # 1->1, 2->2, 3->3, 4->4, padding->0
        behavior_map = {'1': 1, '2': 2, '3': 3, '4': 4}
        seq_behaviors_int = [behavior_map.get(str(b), 0) for b in seq_behaviors]

        # 填充
        padding_len = Config.MAX_SEQ_LEN - len(seq_items)
        if padding_len > 0:
            seq_items = [0] * padding_len + seq_items
            seq_cates = [0] * padding_len + seq_cates
            seq_behaviors_int = [0] * padding_len + seq_behaviors_int

        seq_items_tensor = torch.tensor(seq_items, dtype=torch.long)
        seq_cates_tensor = torch.tensor(seq_cates, dtype=torch.long)
        seq_behaviors_tensor = torch.tensor(seq_behaviors_int, dtype=torch.long)  # 现在是int
        seq_len_tensor = torch.tensor(seq_len, dtype=torch.long)

        label = torch.tensor(row['label'], dtype=torch.float32)

        return (user_id, target_item, target_cate,
                seq_items_tensor, seq_cates_tensor, seq_behaviors_tensor, seq_len_tensor,
                label)


# ==================== 3. DIN模型定义 ====================
class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, units=[128, 64]):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Sequential()
        input_dim = embedding_dim * 4

        for i, unit in enumerate(units):
            self.fc.add_module(f'fc_{i}', nn.Linear(input_dim, unit))
            self.fc.add_module(f'relu_{i}', nn.ReLU())
            self.fc.add_module(f'dropout_{i}', nn.Dropout(Config.DROPOUT_RATE))
            input_dim = unit

        self.fc.add_module('output', nn.Linear(input_dim, 1))

    def forward(self, query, keys, keys_length):
        batch_size, seq_len, emb_dim = keys.size()
        query = query.unsqueeze(1).expand(-1, seq_len, -1)

        all_features = torch.cat([
            query, keys, query * keys, query - keys
        ], dim=-1)

        attn_scores = self.fc(all_features).squeeze(-1)

        mask = torch.arange(seq_len, device=keys.device).unsqueeze(0).expand(batch_size, -1)
        mask = mask < keys_length.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=1)
        output = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)

        return output, attn_weights


class DIN(nn.Module):
    def __init__(self, n_users, n_items, n_cates, embedding_dim=64):
        super(DIN, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(n_cates, embedding_dim, padding_idx=0)

        # 行为类型Embedding：0(padding), 1(浏览), 2(收藏), 3(加购), 4(购买)
        self.behavior_emb = nn.Embedding(5, embedding_dim, padding_idx=0)

        self.attention = AttentionLayer(embedding_dim, Config.ATTENTION_UNITS)

        input_dim = embedding_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, user_ids, target_items, target_cates,
                seq_items, seq_cates, seq_behaviors, seq_lengths):
        # seq_behaviors现在是int类型 (1,2,3,4)
        user_vec = self.user_emb(user_ids)
        target_item_vec = self.item_emb(target_items)
        target_cate_vec = self.cate_emb(target_cates)

        seq_item_vec = self.item_emb(seq_items)
        seq_cate_vec = self.cate_emb(seq_cates)
        seq_behavior_vec = self.behavior_emb(seq_behaviors)  # 传入int索引

        # 历史行为表示
        seq_vec = seq_item_vec + seq_cate_vec + seq_behavior_vec

        interest_vec, attn_weights = self.attention(target_item_vec, seq_vec, seq_lengths)

        concat_vec = torch.cat([user_vec, target_item_vec, target_cate_vec, interest_vec], dim=-1)
        output = self.mlp(concat_vec).squeeze(-1)

        return output, attn_weights


# ==================== 4. 训练流程 ====================
def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.L2_REG)
    criterion = nn.BCELoss()

    best_auc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            (user_ids, target_items, target_cates,
             seq_items, seq_cates, seq_behaviors, seq_lengths,
             labels) = [b.to(Config.DEVICE) for b in batch]

            optimizer.zero_grad()
            preds, _ = model(user_ids, target_items, target_cates,
                             seq_items, seq_cates, seq_behaviors, seq_lengths)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                (user_ids, target_items, target_cates,
                 seq_items, seq_cates, seq_behaviors, seq_lengths,
                 labels) = [b.to(Config.DEVICE) for b in batch]

                preds, _ = model(user_ids, target_items, target_cates,
                                 seq_items, seq_cates, seq_behaviors, seq_lengths)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, 'din_best.pth'))
            print(f"  -> 保存最佳模型 (AUC={val_auc:.4f})")

    return best_auc


# ==================== 5. 主流程 ====================
def main():
    preprocessor = DataPreprocessor(Config.DATA_PATH)
    df = preprocessor.load_and_preprocess()
    samples_df = preprocessor.construct_sequences(df)

    # 划分数据集（时间顺序）
    split_idx = int(len(samples_df) * 0.8)
    train_df = samples_df.iloc[:split_idx]
    val_df = samples_df.iloc[split_idx:]

    print(f"训练集: {len(train_df):,}, 验证集: {len(val_df):,}")

    train_dataset = DINDataset(train_df)
    val_dataset = DINDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    n_users = len(preprocessor.user_encoder.classes_)
    n_items = len(preprocessor.item_encoder.classes_)
    n_cates = len(preprocessor.cate_encoder.classes_)

    print(f"用户数: {n_users:,}, 商品数: {n_items:,}, 类目数: {n_cates:,}")

    model = DIN(n_users, n_items, n_cates, Config.EMBEDDING_DIM).to(Config.DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    best_auc = train_model(model, train_loader, val_loader, Config.EPOCHS)
    print(f"\n训练完成！最佳验证AUC: {best_auc:.4f}")

    # 保存编码器
    import joblib
    joblib.dump(preprocessor.user_encoder, os.path.join(Config.SAVE_DIR, 'user_encoder.pkl'))
    joblib.dump(preprocessor.item_encoder, os.path.join(Config.SAVE_DIR, 'item_encoder.pkl'))
    joblib.dump(preprocessor.cate_encoder, os.path.join(Config.SAVE_DIR, 'cate_encoder.pkl'))
    print(f"编码器已保存到: {Config.SAVE_DIR}")


if __name__ == "__main__":
    main()

# E:\PycharmeProjects\taobao_user_behavior_analysis\venv\Scripts\python.exe E:\PycharmeProjects\taobao_user_behavior_analysis\scripts\DIN.py
# 使用设备: cpu
# 加载数据...
# 原始数据量: 12,253,262条
# 行为类型分布: ['1' '2' '3' '4']
# behavior_type
# 1    11550581
# 3      342171
# 2      242556
# 4      117954
# Name: count, dtype: int64
# 编码ID...
# 构造行为序列...
# 处理用户: 100%|██████████| 10000/10000 [01:02<00:00, 160.06it/s]
# 构造样本数: 233,172
# 正样本: 117,834
# 负样本: 115,338
# 训练集: 186,537, 验证集: 46,635
# 用户数: 10,000, 商品数: 2,876,947, 类目数: 8,916
# 模型参数量: 185,475,586
# Epoch 1/10: 100%|██████████| 729/729 [12:58<00:00,  1.07s/it]
# Epoch 1: Loss=0.6506, Val AUC=0.6780
# Epoch 2/10:   0%|          | 0/729 [00:00<?, ?it/s]  -> 保存最佳模型 (AUC=0.6780)
# Epoch 2/10: 100%|██████████| 729/729 [13:39<00:00,  1.12s/it]
# Epoch 2: Loss=0.6314, Val AUC=0.6842
#   -> 保存最佳模型 (AUC=0.6842)
# Epoch 3/10: 100%|██████████| 729/729 [21:13<00:00,  1.75s/it]
# Epoch 3: Loss=0.6227, Val AUC=0.7052
#   -> 保存最佳模型 (AUC=0.7052)
# Epoch 4/10: 100%|██████████| 729/729 [21:46<00:00,  1.79s/it]
# Epoch 4: Loss=0.6044, Val AUC=0.7191
# Epoch 5/10:   0%|          | 0/729 [00:00<?, ?it/s]  -> 保存最佳模型 (AUC=0.7191)
# Epoch 5/10: 100%|██████████| 729/729 [17:32<00:00,  1.44s/it]
# Epoch 5: Loss=0.5454, Val AUC=0.7205
#   -> 保存最佳模型 (AUC=0.7205)
# Epoch 6/10: 100%|██████████| 729/729 [14:53<00:00,  1.23s/it]
# Epoch 7/10:   0%|          | 0/729 [00:00<?, ?it/s]Epoch 6: Loss=0.3376, Val AUC=0.7031
# Epoch 7/10: 100%|██████████| 729/729 [14:16<00:00,  1.18s/it]
# Epoch 8/10:   0%|          | 0/729 [00:00<?, ?it/s]Epoch 7: Loss=0.2915, Val AUC=0.7035
# Epoch 8/10: 100%|██████████| 729/729 [13:01<00:00,  1.07s/it]
# Epoch 9/10:   0%|          | 0/729 [00:00<?, ?it/s]Epoch 8: Loss=0.2709, Val AUC=0.6968
# Epoch 9/10: 100%|██████████| 729/729 [13:12<00:00,  1.09s/it]
# Epoch 9: Loss=0.2495, Val AUC=0.6947
# Epoch 10/10: 100%|██████████| 729/729 [12:39<00:00,  1.04s/it]
# Epoch 10: Loss=0.2306, Val AUC=0.6902
#
# 训练完成！最佳验证AUC: 0.7205
# 编码器已保存到: E:\PycharmeProjects\taobao_user_behavior_analysis\results\din_model
#
# Process finished with exit code 0
