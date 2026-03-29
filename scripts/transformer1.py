"""
SASRec - Self-Attentive Sequential Recommendation
修复版：解决NaN问题，增加数值稳定性检查
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
from collections import defaultdict

warnings.filterwarnings('ignore')


class Config:
    """保守配置 - 防止NaN"""
    DATA_PATH = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet'
    SAVE_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\sasrec'
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_START = '2014-11-19'
    TEST_END = '2014-12-18'

    MAX_SEQ_LEN = 30
    MIN_ITEM_FREQ = 5

    # 更保守的模型配置
    EMBED_DIM = 32
    NUM_BLOCKS = 2
    NUM_HEADS = 2
    DROPOUT_RATE = 0.5
    MAX_LEN = 30

    # 保守的训练配置（防止NaN）
    BATCH_SIZE = 256
    LR = 0.0005  # 降低学习率（原来是0.001）
    WEIGHT_DECAY = 5e-4  # 降低weight decay（原来是1e-3）
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 3
    NEGATIVE_RATIO = 3

    # 梯度裁剪更严格
    MAX_GRAD_NORM = 0.5  # 从1.0降到0.5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DataPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.item2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2item = {0: '<PAD>', 1: '<UNK>'}
        self.user_histories = {}

    def load_and_filter(self):
        print(f"[1/4] 加载数据: {self.cfg.DATA_PATH}")
        df = pd.read_parquet(self.cfg.DATA_PATH)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        else:
            df['date'] = pd.to_datetime(df['datetime']).dt.date

        train_start_date = pd.to_datetime(self.cfg.TRAIN_START).date()
        test_end_date = pd.to_datetime(self.cfg.TEST_END).date()

        mask = (df['date'] >= train_start_date) & (df['date'] <= test_end_date)
        df = df[mask].copy()
        print(f"      过滤后数据: {len(df):,}条")

        # 检查数据完整性
        assert not df['item_id'].isna().any(), "item_id包含NaN"
        assert not df['user_id'].isna().any(), "user_id包含NaN"

        print("[2/4] 统计商品频次...")
        item_counts = df['item_id'].value_counts()
        valid_items = set(item_counts[item_counts >= self.cfg.MIN_ITEM_FREQ].index)
        print(f"      高频商品: {len(valid_items):,} / {len(item_counts):,}")

        for item in valid_items:
            idx = len(self.item2idx)
            self.item2idx[int(item)] = idx
            self.idx2item[idx] = int(item)

        self.num_items = len(self.item2idx)
        print(f"      商品索引大小: {self.num_items}")

        print("[3/4] 构建用户行为序列...")
        df_sorted = df.sort_values(['user_id', 'datetime'])

        for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted)):
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            cate = int(row['item_category'])
            behavior = int(row['behavior_type']) if isinstance(row['behavior_type'], (int, float)) else int(
                row['behavior_type'])

            item_idx = self.item2idx.get(item_id, 1)

            if user_id not in self.user_histories:
                self.user_histories[user_id] = []

            self.user_histories[user_id].append({
                'item_idx': item_idx,
                'item_id': item_id,
                'cate': cate,
                'behavior': behavior,
                'datetime': row['datetime'] if 'datetime' in row else pd.to_datetime(row['time'])
            })

        print(f"      用户数量: {len(self.user_histories):,}")
        return df

    def generate_sequences(self):
        print("[4/4] 生成序列样本...")

        train_samples, val_samples, test_samples = [], [], []

        for user_id, hist in tqdm(self.user_histories.items(), desc="构造样本"):
            if len(hist) < 2:
                continue

            for i in range(1, len(hist)):
                target = hist[i]
                target_item = target['item_idx']
                target_behavior = target['behavior']

                start_idx = max(0, i - self.cfg.MAX_SEQ_LEN)
                history = hist[start_idx:i]

                seq_items = [h['item_idx'] for h in history]
                seq_cates = [h['cate'] for h in history]

                if len(seq_items) < self.cfg.MAX_SEQ_LEN:
                    pad_len = self.cfg.MAX_SEQ_LEN - len(seq_items)
                    seq_items = [0] * pad_len + seq_items
                    seq_cates = [0] * pad_len + seq_cates

                label = 1 if target_behavior == 4 else 0

                sample = {
                    'user_id': user_id,
                    'seq_items': seq_items,
                    'seq_cates': seq_cates,
                    'target_item': target_item,
                    'target_cate': target['cate'],
                    'seq_len': min(i, self.cfg.MAX_SEQ_LEN),
                    'label': label
                }

                if i < len(hist) * 0.7:
                    train_samples.append(sample)
                elif i < len(hist) * 0.8:
                    val_samples.append(sample)
                else:
                    test_samples.append(sample)

        print(f"      Train: {len(train_samples):,} | Val: {len(val_samples):,} | Test: {len(test_samples):,}")
        return train_samples, val_samples, test_samples

    def add_negative_samples(self, samples, split_name=""):
        print(f"[5/5] 负采样 {split_name}...")

        user_purchases = defaultdict(set)
        user_views = defaultdict(set)

        for s in samples:
            uid = s['user_id']
            if s['label'] == 1:
                user_purchases[uid].add(s['target_item'])
            user_views[uid].add(s['target_item'])

        enhanced_samples = []

        for s in tqdm(samples, desc="负采样"):
            enhanced_samples.append(s)

            if s['label'] == 1:
                uid = s['user_id']
                candidates = list(user_views[uid] - user_purchases[uid])

                if len(candidates) == 0:
                    candidates = list(set(range(2, self.num_items)) - user_purchases[uid])

                n_neg = min(self.cfg.NEGATIVE_RATIO, len(candidates))
                if n_neg > 0:
                    neg_items = np.random.choice(candidates, size=n_neg, replace=False)
                    for neg_item in neg_items:
                        neg_sample = s.copy()
                        neg_sample['target_item'] = int(neg_item)
                        neg_sample['label'] = 0
                        enhanced_samples.append(neg_sample)

        pos_after = sum(s['label'] for s in enhanced_samples)
        print(
            f"      采样后: {len(enhanced_samples):,} (正: {pos_after:,}, 正样本率: {pos_after / len(enhanced_samples) * 100:.2f}%)")
        return enhanced_samples


class SASRecDataset(Dataset):
    def __init__(self, samples, num_items):
        self.samples = samples
        self.num_items = num_items

        # 检查数据有效性
        for i, s in enumerate(self.samples):
            assert all(isinstance(x, int) for x in s['seq_items']), f"样本{i} seq_items包含非整数"
            assert s['label'] in [0, 1], f"样本{i} label不在[0,1]中"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'seq_items': torch.tensor(s['seq_items'], dtype=torch.long),
            'seq_cates': torch.tensor(s['seq_cates'], dtype=torch.long),
            'target_item': torch.tensor(s['target_item'], dtype=torch.long),
            'target_cate': torch.tensor(s['target_cate'], dtype=torch.long),
            'seq_len': torch.tensor(s['seq_len'], dtype=torch.long),
            'label': torch.tensor(s['label'], dtype=torch.float)
        }


class SASRec(nn.Module):
    def __init__(self, num_items, num_cates, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_items = num_items

        # 使用更保守的初始化
        self.item_emb = nn.Embedding(num_items, cfg.EMBED_DIM, padding_idx=0)
        self.cate_emb = nn.Embedding(num_cates + 1, cfg.EMBED_DIM, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.MAX_LEN + 1, cfg.EMBED_DIM)

        self.dropout = nn.Dropout(cfg.DROPOUT_RATE)

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=cfg.EMBED_DIM,
                num_heads=cfg.NUM_HEADS,
                dropout=cfg.DROPOUT_RATE,
                batch_first=True
            ) for _ in range(cfg.NUM_BLOCKS)
        ])

        self.forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM * 4),
                nn.ReLU(),
                nn.Dropout(cfg.DROPOUT_RATE),
                nn.Linear(cfg.EMBED_DIM * 4, cfg.EMBED_DIM)
            ) for _ in range(cfg.NUM_BLOCKS)
        ])

        self.layer_norms_attn = nn.ModuleList([nn.LayerNorm(cfg.EMBED_DIM) for _ in range(cfg.NUM_BLOCKS)])
        self.layer_norms_forward = nn.ModuleList([nn.LayerNorm(cfg.EMBED_DIM) for _ in range(cfg.NUM_BLOCKS)])

        self.output_bias = nn.Parameter(torch.zeros(num_items))

        self._init_weights()

        print(f"[模型] SASRec | 参数量: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def _init_weights(self):
        # 更保守的初始化（防止初始值过大）
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)  # 从0.01改为0.02，但限制范围
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # 添加gain控制
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, seq_items, seq_cates, target_item, seq_len, return_logits=False):
        batch_size = seq_items.size(0)
        seq_len_size = seq_items.size(1)

        # 检查输入有效性
        if torch.isnan(seq_items).any() or torch.isinf(seq_items).any():
            raise ValueError("输入seq_items包含NaN或Inf")

        padding_mask = (seq_items == 0)
        causal_mask = torch.triu(torch.ones(seq_len_size, seq_len_size), diagonal=1).bool().to(self.cfg.DEVICE)

        # Embedding
        item_e = self.item_emb(seq_items)
        cate_e = self.cate_emb(seq_cates)

        positions = torch.arange(seq_len_size, device=self.cfg.DEVICE).unsqueeze(0).expand(batch_size, -1)
        positions = positions * (seq_items != 0).long()
        pos_e = self.pos_emb(positions)

        x = item_e + cate_e + pos_e
        x = self.dropout(x)

        # Transformer Blocks
        for i in range(self.cfg.NUM_BLOCKS):
            x_norm = self.layer_norms_attn[i](x)
            attn_out, _ = self.attention_layers[i](
                x_norm, x_norm, x_norm,
                key_padding_mask=padding_mask,
                attn_mask=causal_mask
            )
            x = x + attn_out

            x_norm = self.layer_norms_forward[i](x)
            ff_out = self.forward_layers[i](x_norm)
            x = x + ff_out

            # 检查每层的输出
            if torch.isnan(x).any():
                raise ValueError(f"Transformer第{i}层输出NaN")

        user_repr = x[torch.arange(batch_size), seq_len - 1]
        target_emb = self.item_emb(target_item)

        # 内积前做L2归一化（防止数值过大）
        user_repr = nn.functional.normalize(user_repr, p=2, dim=1)
        target_emb = nn.functional.normalize(target_emb, p=2, dim=1)

        logits = (user_repr * target_emb).sum(dim=-1)
        logits = logits + self.output_bias[target_item]

        # 限制logits范围（防止sigmoid前爆炸）
        logits = torch.clamp(logits, min=-10, max=10)

        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)


def train_epoch(model, train_loader, optimizer, criterion, cfg, epoch):
    """带NaN检测的训练"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    nan_count = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")

    for batch_idx, batch in enumerate(pbar):
        try:
            seq_items = batch['seq_items'].to(cfg.DEVICE)
            seq_cates = batch['seq_cates'].to(cfg.DEVICE)
            target_item = batch['target_item'].to(cfg.DEVICE)
            seq_len = batch['seq_len'].to(cfg.DEVICE)
            labels = batch['label'].to(cfg.DEVICE)

            optimizer.zero_grad()

            logits = model(seq_items, seq_cates, target_item, seq_len, return_logits=True)

            # 检查logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n[警告] Batch {batch_idx} 产生NaN/Inf logits，跳过")
                nan_count += 1
                continue

            loss = criterion(logits, labels)

            # 检查loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[警告] Batch {batch_idx} 产生NaN/Inf loss，跳过")
                nan_count += 1
                continue

            loss.backward()

            # 梯度裁剪前检查
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.MAX_GRAD_NORM)
            if torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
                print(f"\n[警告] Batch {batch_idx} 梯度爆炸，已裁剪但仍是NaN")
                optimizer.zero_grad()
                nan_count += 1
                continue

            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                # 再次检查
                valid_mask = ~(torch.isnan(probs) | torch.isinf(probs))
                if valid_mask.all():
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'nan_batches': nan_count})

        except Exception as e:
            print(f"\n[错误] Batch {batch_idx} 发生异常: {e}")
            continue

    if len(all_labels) == 0:
        print("[错误] 所有batch都产生NaN，无法计算AUC")
        return float('inf'), 0.5

    avg_loss = total_loss / (len(train_loader) - nan_count) if (len(train_loader) - nan_count) > 0 else float('inf')

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError as e:
        print(f"[错误] 计算AUC失败: {e}")
        auc = 0.5

    return avg_loss, auc


def evaluate(model, data_loader, cfg):
    """带NaN检测的评估"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            try:
                seq_items = batch['seq_items'].to(cfg.DEVICE)
                seq_cates = batch['seq_cates'].to(cfg.DEVICE)
                target_item = batch['target_item'].to(cfg.DEVICE)
                seq_len = batch['seq_len'].to(cfg.DEVICE)
                labels = batch['label'].to(cfg.DEVICE)

                probs = model(seq_items, seq_cates, target_item, seq_len, return_logits=False)

                # 过滤NaN
                valid_mask = ~(torch.isnan(probs) | torch.isinf(probs))
                if valid_mask.any():
                    all_preds.extend(probs[valid_mask].cpu().numpy())
                    all_labels.extend(labels[valid_mask].cpu().numpy())

            except Exception as e:
                print(f"[评估错误]: {e}")
                continue

    if len(all_labels) == 0:
        print("[错误] 评估集全部NaN")
        return 0.5, 0.0

    try:
        auc = roc_auc_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    except ValueError:
        auc, f1 = 0.5, 0.0

    return auc, f1


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    print("=" * 60)
    print("SASRec - 自注意力序列推荐模型 (NaN修复版)")
    print(f"设备: {cfg.DEVICE}")
    print(f"学习率: {cfg.LR} | Weight Decay: {cfg.WEIGHT_DECAY} | Grad Clip: {cfg.MAX_GRAD_NORM}")
    print("=" * 60)

    # 1. 数据预处理
    preprocessor = DataPreprocessor(cfg)
    preprocessor.load_and_filter()

    train_samples, val_samples, test_samples = preprocessor.generate_sequences()

    train_samples = preprocessor.add_negative_samples(train_samples, "训练集")
    val_samples = preprocessor.add_negative_samples(val_samples, "验证集")
    test_samples = preprocessor.add_negative_samples(test_samples, "测试集")

    if len(train_samples) == 0:
        print("错误：没有生成训练样本")
        return

    # 2. 创建DataLoader
    num_cates = max([max(s['seq_cates']) for s in train_samples]) + 1

    train_dataset = SASRecDataset(train_samples, preprocessor.num_items)
    val_dataset = SASRecDataset(val_samples, preprocessor.num_items)
    test_dataset = SASRecDataset(test_samples, preprocessor.num_items)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 初始化模型
    model = SASRec(preprocessor.num_items, num_cates, cfg).to(cfg.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        print("[提示] 当前PyTorch版本不支持scheduler的verbose参数")

    criterion = nn.BCEWithLogitsLoss()

    # 4. 训练循环
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0

    print(f"\n开始训练...")

    for epoch in range(cfg.EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, cfg, epoch)
        val_auc, val_f1 = evaluate(model, val_loader, cfg)

        if np.isnan(train_auc) or np.isnan(val_auc):
            print(f"[致命错误] 第{epoch + 1}轮产生NaN，终止训练")
            break

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"      ↓ 学习率调整: {current_lr:.6f} -> {new_lr:.6f}")

        gap = train_auc - val_auc
        print(f"Epoch {epoch + 1:2d}: Loss={train_loss:.4f} | "
              f"Train={train_auc:.4f} | Val={val_auc:.4f} | F1={val_f1:.4f} | Gap={gap:.4f}")

        # 保存检查点
        if val_auc > best_val_auc and not np.isnan(val_auc):
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"      ✓ 新的最佳模型 (Val AUC: {val_auc:.4f})")

            # 及时保存，防止崩溃丢失
            torch.save(best_model_state, os.path.join(cfg.SAVE_DIR, 'best_model_checkpoint.pth'))
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"\n[!] 早停，最佳Val AUC: {best_val_auc:.4f}")
                break

    # 5. 测试集评估
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(cfg.DEVICE)

    test_auc, test_f1 = evaluate(model, test_loader, cfg)

    print(f"\n{'=' * 60}")
    print(f"最终结果:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test F1:  {test_f1:.4f}")
    print(f"对比MLP(0.6061): 提升 {test_auc - 0.6061:+.4f}")
    print(f"{'=' * 60}")

    # 保存结果
    results = {
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1),
        'num_items': preprocessor.num_items,
        'embed_dim': cfg.EMBED_DIM,
        'config': {k: str(v) if isinstance(v, torch.device) else v for k, v in cfg.__dict__.items() if
                   not k.startswith('_')}
    }
    import json
    with open(os.path.join(cfg.SAVE_DIR, 'sasrec_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'sasrec_model.pth'))
    print(f"\n结果已保存至: {cfg.SAVE_DIR}")


if __name__ == "__main__":
    main()

# 最终结果:
# Test AUC: 0.6379
# Test F1:  0.0289
# 对比MLP(0.6061): 提升 +0.0318