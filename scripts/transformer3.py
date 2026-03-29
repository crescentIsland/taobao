"""
SASRec - 基础修复版 (解决第一轮过拟合 + Train AUC显示Bug)
核心修复:
1. 标签平滑后的AUC计算逻辑修正
2. 学习率预热(Warmup) + 余弦退火，避免初期跳跃
3. Model EMA(指数移动平均)稳定训练
4. 早停阈值调整为2轮（快速收敛场景）
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from tqdm import tqdm
import warnings
from collections import defaultdict
import json

warnings.filterwarnings('ignore')


class Config:
    """基础修复配置"""
    DATA_PATH = r'E:\PycharmeProjects\taobao_user_behavior_analysis\data\processed\user_action_processed.parquet'
    SAVE_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\sasrec_fixed'
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_START = '2014-11-19'
    TEST_END = '2014-12-18'

    MAX_SEQ_LEN = 30
    MIN_ITEM_FREQ = 5

    EMBED_DIM = 64
    NUM_BLOCKS = 4
    NUM_HEADS = 4
    DROPOUT_RATE = 0.3
    MAX_LEN = 30

    BATCH_SIZE = 256
    # 关键修改1: 降低初始学习率，防止第一轮跳跃过拟合
    LR = 0.0005  # 从0.001降至0.0005
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    # 关键修改2: 降低早停轮数（因为收敛快）
    EARLY_STOPPING_PATIENCE = 2  # 从5降至2
    MAX_GRAD_NORM = 1.0

    TARGET_POS_RATIO = 0.25
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1

    # 关键修改3: 学习率调度参数
    WARMUP_EPOCHS = 2  # 新增: 前2轮线性预热，避免初期震荡
    T_0 = 10  # 余弦周期
    T_MULT = 2

    # 关键修改4: EMA参数
    EMA_DECAY = 0.999  # 指数移动平均衰减率

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42


class EMA:
    """指数移动平均 - 用于稳定训练，减少过拟合"""

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        weight = alpha_t * torch.pow(1. - p_t, self.gamma)
        loss = weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DataPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.item2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2item = {0: '<PAD>', 1: '<UNK>'}
        self.user_histories = {}
        self.pos_weight = 1.0

    def load_and_filter(self):
        print(f"[1/5] 加载数据: {self.cfg.DATA_PATH}")
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

        assert not df['item_id'].isna().any(), "item_id包含NaN"
        assert not df['user_id'].isna().any(), "user_id包含NaN"

        behavior_col = 'behavior_type'
        if behavior_col in df.columns:
            pos_rate = (df[behavior_col] == 4).mean()
            print(f"      原始正样本率: {pos_rate:.4%}")
            self.pos_weight = min((1 - pos_rate) / (pos_rate + 1e-8), 100.0)
            print(f"      自动计算类别权重: {self.pos_weight:.2f}")

        print("[2/5] 统计商品频次...")
        item_counts = df['item_id'].value_counts()
        valid_items = set(item_counts[item_counts >= self.cfg.MIN_ITEM_FREQ].index)
        print(f"      高频商品: {len(valid_items):,} / {len(item_counts):,}")

        for item in valid_items:
            idx = len(self.item2idx)
            self.item2idx[int(item)] = idx
            self.idx2item[idx] = int(item)

        self.num_items = len(self.item2idx)
        print(f"      商品索引大小: {self.num_items}")

        print("[3/5] 构建用户行为序列...")
        df_sorted = df.sort_values(['user_id', 'datetime'])

        for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="构建序列"):
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            cate = int(row['item_category'])

            behavior_val = row['behavior_type']
            if isinstance(behavior_val, (int, float)):
                behavior = int(behavior_val)
            else:
                behavior = int(behavior_val)

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
        print("[4/5] 生成序列样本...")

        train_samples, val_samples, test_samples = [], [], []
        pos_count = 0

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
                if label == 1:
                    pos_count += 1

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

        print(f"      原始正样本: {pos_count:,}")
        print(f"      划分: Train {len(train_samples):,} | Val {len(val_samples):,} | Test {len(test_samples):,}")
        return train_samples, val_samples, test_samples

    def add_negative_samples_dynamic(self, samples, split_name=""):
        print(f"[5/5] 动态负采样 {split_name}...")

        pos_samples = [s for s in samples if s['label'] == 1]
        neg_samples = [s for s in samples if s['label'] == 0]

        n_pos = len(pos_samples)
        n_neg = len(neg_samples)

        if n_pos == 0:
            print(f"      警告: {split_name} 无正样本")
            return samples

        target_neg_count = int(n_pos * (1 - self.cfg.TARGET_POS_RATIO) / self.cfg.TARGET_POS_RATIO)

        print(f"      原始: {n_pos}正/{n_neg}负 ({n_pos / max(1, n_pos + n_neg):.2%})")
        print(f"      目标: {n_pos}正/{target_neg_count}负 ({self.cfg.TARGET_POS_RATIO:.1%})")

        user_interacted = defaultdict(set)
        for s in samples:
            user_interacted[s['user_id']].add(s['target_item'])

        enhanced_samples = pos_samples.copy()

        if n_neg > target_neg_count:
            np.random.shuffle(neg_samples)
            selected_neg = neg_samples[:target_neg_count]
            enhanced_samples.extend(selected_neg)
        else:
            enhanced_samples.extend(neg_samples)
            need_more = target_neg_count - n_neg
            neg_per_pos = max(1, need_more // n_pos)

            for pos_s in tqdm(pos_samples, desc="生成额外负样本"):
                uid = pos_s['user_id']
                candidates = list(set(range(2, self.num_items)) - user_interacted[uid])

                if len(candidates) == 0:
                    continue

                n_sample = min(neg_per_pos, len(candidates))
                neg_items = np.random.choice(candidates, size=n_sample, replace=False)

                for neg_item in neg_items:
                    if len(enhanced_samples) >= n_pos + target_neg_count:
                        break

                    neg_sample = pos_s.copy()
                    neg_sample['target_item'] = int(neg_item)
                    neg_sample['label'] = 0
                    enhanced_samples.append(neg_sample)

                if len(enhanced_samples) >= n_pos + target_neg_count:
                    break

        np.random.shuffle(enhanced_samples)

        final_pos = sum(s['label'] for s in enhanced_samples)
        final_total = len(enhanced_samples)
        final_ratio = final_pos / final_total if final_total > 0 else 0

        print(f"      采样后: {final_pos}正/{final_total - final_pos}负 ({final_ratio:.2%})")

        return enhanced_samples


class SASRecDataset(Dataset):
    def __init__(self, samples, num_items, label_smoothing=0.0):
        self.samples = samples
        self.num_items = num_items
        self.label_smoothing = label_smoothing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        label = s['label']
        if self.label_smoothing > 0:
            label = label * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        return {
            'seq_items': torch.tensor(s['seq_items'], dtype=torch.long),
            'seq_cates': torch.tensor(s['seq_cates'], dtype=torch.long),
            'target_item': torch.tensor(s['target_item'], dtype=torch.long),
            'target_cate': torch.tensor(s['target_cate'], dtype=torch.long),
            'seq_len': torch.tensor(s['seq_len'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


class SASRec(nn.Module):
    def __init__(self, num_items, num_cates, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_items = num_items

        self.item_emb = nn.Embedding(num_items, cfg.EMBED_DIM, padding_idx=0)
        self.cate_emb = nn.Embedding(num_cates + 1, cfg.EMBED_DIM, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.MAX_LEN + 1, cfg.EMBED_DIM)

        self.dropout = nn.Dropout(cfg.DROPOUT_RATE)
        self.emb_dropout = nn.Dropout(cfg.DROPOUT_RATE * 0.5)

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
                nn.GELU(),
                nn.Dropout(cfg.DROPOUT_RATE),
                nn.Linear(cfg.EMBED_DIM * 4, cfg.EMBED_DIM)
            ) for _ in range(cfg.NUM_BLOCKS)
        ])

        self.layer_norms_attn = nn.ModuleList([nn.LayerNorm(cfg.EMBED_DIM) for _ in range(cfg.NUM_BLOCKS)])
        self.layer_norms_forward = nn.ModuleList([nn.LayerNorm(cfg.EMBED_DIM) for _ in range(cfg.NUM_BLOCKS)])
        self.final_norm = nn.LayerNorm(cfg.EMBED_DIM)

        self.output_bias = nn.Parameter(torch.zeros(num_items))

        self.prediction_head = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM * 2, cfg.EMBED_DIM),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(cfg.EMBED_DIM, 1)
        )

        self._init_weights()

        print(
            f"[模型] SASRec修复版 | 维度{cfg.EMBED_DIM} | 层数{cfg.NUM_BLOCKS} | 参数量: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, seq_items, seq_cates, target_item, seq_len, return_logits=False):
        batch_size = seq_items.size(0)
        seq_len_size = seq_items.size(1)

        padding_mask = (seq_items == 0)
        causal_mask = torch.triu(torch.ones(seq_len_size, seq_len_size), diagonal=1).bool().to(self.cfg.DEVICE)

        item_e = self.item_emb(seq_items)
        cate_e = self.cate_emb(seq_cates)
        positions = torch.arange(seq_len_size, device=self.cfg.DEVICE).unsqueeze(0).expand(batch_size, -1)
        positions = positions * (seq_items != 0).long()
        pos_e = self.pos_emb(positions)

        x = item_e + cate_e + pos_e
        x = self.emb_dropout(x)

        for i in range(self.cfg.NUM_BLOCKS):
            x_norm = self.layer_norms_attn[i](x)
            attn_out, _ = self.attention_layers[i](
                x_norm, x_norm, x_norm,
                key_padding_mask=padding_mask,
                attn_mask=causal_mask,
                need_weights=False
            )
            x = x + attn_out

            x_norm = self.layer_norms_forward[i](x)
            ff_out = self.forward_layers[i](x_norm)
            x = x + ff_out

        x = self.final_norm(x)
        user_repr = x[torch.arange(batch_size), seq_len - 1]
        target_emb = self.item_emb(target_item)

        combined = torch.cat([user_repr, target_emb], dim=-1)
        logits = self.prediction_head(combined).squeeze(-1)
        logits = logits + self.output_bias[target_item]

        if return_logits:
            return logits
        return torch.sigmoid(logits)


def train_epoch(model, train_loader, optimizer, criterion, cfg, epoch, ema=None):
    """关键修复: 修正标签平滑后的AUC计算"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    nan_count = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        try:
            seq_items = batch['seq_items'].to(cfg.DEVICE)
            seq_cates = batch['seq_cates'].to(cfg.DEVICE)
            target_item = batch['target_item'].to(cfg.DEVICE)
            seq_len = batch['seq_len'].to(cfg.DEVICE)
            labels = batch['label'].to(cfg.DEVICE)  # 注意: 这里已经是平滑后的标签(0.05或0.95)

            optimizer.zero_grad()

            logits = model(seq_items, seq_cates, target_item, seq_len, return_logits=True)

            if torch.isnan(logits).any():
                nan_count += 1
                continue

            # 计算Loss使用平滑标签
            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.MAX_GRAD_NORM)
            optimizer.step()

            # 更新EMA
            if ema is not None:
                ema.update()

            total_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                valid_mask = ~(torch.isnan(probs) | torch.isinf(probs))
                if valid_mask.all():
                    all_preds.extend(probs.cpu().numpy())
                    # 关键修复: 将平滑标签转回0/1用于AUC计算
                    # 平滑标签0.05对应原标签0，0.95对应原标签1
                    original_labels = (labels.cpu().numpy() > 0.5).astype(int)
                    all_labels.extend(original_labels)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'nan': nan_count,
                'pos_rate': f'{labels.mean().cpu().item():.2%}'
            })

        except Exception as e:
            print(f"\n[错误] Batch {batch_idx}: {e}")
            continue

    if len(all_labels) == 0:
        return float('inf'), 0.5

    avg_loss = total_loss / max(1, len(train_loader) - nan_count)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception as e:
        print(f"[AUC计算错误]: {e}")
        auc = 0.5

    return avg_loss, auc


def evaluate(model, data_loader, cfg, find_threshold=True, use_ema=False, ema=None):
    """评估函数，支持EMA模型评估"""
    # 如果使用EMA，先应用shadow权重
    if use_ema and ema is not None:
        ema.apply_shadow()
        model.eval()
    else:
        model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估", leave=False):
            try:
                seq_items = batch['seq_items'].to(cfg.DEVICE)
                seq_cates = batch['seq_cates'].to(cfg.DEVICE)
                target_item = batch['target_item'].to(cfg.DEVICE)
                seq_len = batch['seq_len'].to(cfg.DEVICE)
                labels = batch['label'].to(cfg.DEVICE)

                probs = model(seq_items, seq_cates, target_item, seq_len, return_logits=False)

                valid_mask = ~(torch.isnan(probs) | torch.isinf(probs))
                if valid_mask.any():
                    all_preds.extend(probs[valid_mask].cpu().numpy())
                    # 同样处理标签平滑
                    original_labels = (labels[valid_mask].cpu().numpy() > 0.5).astype(int)
                    all_labels.extend(original_labels)

            except Exception as e:
                print(f"[评估错误]: {e}")
                continue

    # 恢复原模型参数
    if use_ema and ema is not None:
        ema.restore()

    if len(all_labels) == 0:
        return 0.5, 0.0, 0.5

    try:
        auc = roc_auc_score(all_labels, all_preds)

        if find_threshold and len(set(all_labels)) > 1:
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1 = f1_scores[best_idx]
        else:
            best_threshold = 0.5
            best_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

    except Exception as e:
        print(f"[指标计算错误]: {e}")
        auc, best_f1, best_threshold = 0.5, 0.0, 0.5

    return auc, best_f1, best_threshold


class WarmupCosineScheduler:
    """学习率预热 + 余弦退火调度器"""

    def __init__(self, optimizer, warmup_epochs, T_0, T_mult, eta_min=1e-6, base_lr=0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lr = base_lr
        self.current_epoch = 0

        # 预热阶段后的余弦调度器
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

    def step(self, epoch=None):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段：线性增加学习率
            warmup_factor = self.current_epoch / self.warmup_epochs
            new_lr = self.eta_min + (self.base_lr - self.eta_min) * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            # 预热后：使用余弦退火
            self.cosine_scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def main():
    cfg = Config()
    set_seed(cfg.SEED)

    print("=" * 70)
    print("SASRec - 基础修复版 (解决第一轮过拟合 + AUC显示Bug)")
    print(f"设备: {cfg.DEVICE}")
    print(f"配置: {cfg.NUM_BLOCKS}层/{cfg.EMBED_DIM}维 | LR={cfg.LR} | Warmup={cfg.WARMUP_EPOCHS}轮")
    print(f"早停耐心: {cfg.EARLY_STOPPING_PATIENCE}轮 | EMA衰减: {cfg.EMA_DECAY}")
    print("=" * 70)

    # 1. 数据预处理
    preprocessor = DataPreprocessor(cfg)
    preprocessor.load_and_filter()

    train_samples, val_samples, test_samples = preprocessor.generate_sequences()

    train_samples = preprocessor.add_negative_samples_dynamic(train_samples, "训练集")
    val_samples = preprocessor.add_negative_samples_dynamic(val_samples, "验证集")
    test_samples = preprocessor.add_negative_samples_dynamic(test_samples, "测试集")

    if len(train_samples) == 0:
        print("错误：没有生成训练样本")
        return

    # 2. 创建DataLoader
    num_cates = max([max(s['seq_cates']) for s in train_samples]) + 1

    train_dataset = SASRecDataset(train_samples, preprocessor.num_items, cfg.LABEL_SMOOTHING)
    val_dataset = SASRecDataset(val_samples, preprocessor.num_items, 0)
    test_dataset = SASRecDataset(test_samples, preprocessor.num_items, 0)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 初始化模型和EMA
    model = SASRec(preprocessor.num_items, num_cates, cfg).to(cfg.DEVICE)
    ema = EMA(model, decay=cfg.EMA_DECAY)  # 初始化EMA

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # 关键修改: 使用预热+余弦退火调度器
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.WARMUP_EPOCHS,
        T_0=cfg.T_0,
        T_mult=cfg.T_MULT,
        eta_min=1e-6,
        base_lr=cfg.LR
    )

    criterion = FocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)

    # 4. 训练循环
    best_val_auc = 0
    best_model_state = None
    best_ema_state = None
    patience_counter = 0
    threshold = 0.5

    # 记录每轮结果
    history = []

    print(f"\n开始训练...")

    for epoch in range(cfg.EPOCHS):
        # 训练阶段
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, cfg, epoch, ema)

        # 分别评估普通模型和EMA模型
        val_auc, val_f1, _ = evaluate(model, val_loader, cfg, find_threshold=False, use_ema=False)
        val_auc_ema, val_f1_ema, threshold_ema = evaluate(model, val_loader, cfg, find_threshold=True, use_ema=True,
                                                          ema=ema)

        # 使用更好的那个结果
        use_ema = val_auc_ema > val_auc
        best_val_current = max(val_auc, val_auc_ema)
        best_f1_current = val_f1_ema if use_ema else val_f1

        scheduler.step()
        current_lr = scheduler.get_lr()

        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_auc_ema': val_auc_ema,
            'val_f1': best_f1_current,
            'lr': current_lr,
            'gap': train_auc - best_val_current
        })

        print(f"Epoch {epoch + 1:2d}: Loss={train_loss:.4f} | "
              f"Train={train_auc:.4f} | Val={best_val_current:.4f} ({'EMA' if use_ema else 'Raw'}) | "
              f"F1={best_f1_current:.4f} | LR={current_lr:.6f}")

        # 保存最佳模型（优先保存EMA版本，更稳定）
        if best_val_current > best_val_auc and not np.isnan(best_val_current):
            best_val_auc = best_val_current

            if use_ema:
                # 保存EMA的shadow参数
                ema.apply_shadow()
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_ema_state = 'ema_used'
                ema.restore()
            else:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_ema_state = 'raw_used'

            patience_counter = 0
            print(f"      ✓ 新的最佳模型 (Val AUC: {best_val_current:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"\n[!] 早停，最佳Val AUC: {best_val_auc:.4f}")
                break

    # 5. 测试集评估
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(cfg.DEVICE)

    test_auc, test_f1, test_threshold = evaluate(model, test_loader, cfg, find_threshold=True)

    # 同时测试EMA版本
    test_auc_ema, test_f1_ema, _ = evaluate(model, test_loader, cfg, find_threshold=True, use_ema=True, ema=ema)

    print(f"\n{'=' * 70}")
    print(f"最终结果:")
    print(f"Test AUC (Best Model): {test_auc:.4f} (F1={test_f1:.4f})")
    if best_ema_state == 'ema_used':
        print(f"Test AUC (EMA Model):  {test_auc_ema:.4f} (F1={test_f1_ema:.4f})")
    print(f"对比上次: 0.7477 -> {max(test_auc, test_auc_ema):.4f} ({max(test_auc, test_auc_ema) - 0.7477:+.4f})")
    print(f"{'=' * 70}")

    # 保存完整结果
    results = {
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1),
        'test_auc_ema': float(test_auc_ema) if best_ema_state == 'ema_used' else None,
        'history': history,
        'config': {k: str(v) if isinstance(v, torch.device) else v for k, v in cfg.__dict__.items() if
                   not k.startswith('_')}
    }

    with open(os.path.join(cfg.SAVE_DIR, 'fixed_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'final_model.pth'))
    print(f"\n结果已保存至: {cfg.SAVE_DIR}")


if __name__ == "__main__":
    main()

# [!] 早停，最佳Val AUC: 0.7604
#
# ======================================================================
# 最终结果:
# Test AUC (Best Model): 0.7525 (F1=0.5215)
# Test AUC (EMA Model):  0.7027 (F1=0.4857)
# 对比上次: 0.7477 -> 0.7525 (+0.0048)