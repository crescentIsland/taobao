# 完整版：SASRec 误差分析 + 性能测试（纯 CPU 环境）
# 直接复制到你的项目中，替换原有的相关部分

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import torch
import torch.nn.functional as F
from collections import defaultdict
import time
import psutil
import json
import os


class SASRecErrorAnalyzer:
    """SASRec 误差分析器 - 解决【待考证】误差分析问题"""

    def __init__(self, model, cfg, item_popularity=None):
        self.model = model
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.item_popularity = item_popularity or {}
        self.analysis_results = {}

    def extract_sample_features(self, dataloader, max_samples=5000):
        """提取样本特征：用于聚类和分布分析"""
        self.model.eval()
        samples_data = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(samples_data) >= max_samples:
                    break

                seq_items = batch['seq_items'].to(self.device)
                seq_cates = batch['seq_cates'].to(self.device)
                target_item = batch['target_item'].to(self.device)
                seq_len = batch['seq_len'].to(self.device)
                labels = batch['label'].to(self.device)

                # 获取预测概率
                probs = self.model(seq_items, seq_cates, target_item, seq_len, return_logits=False)

                # 获取中间表示
                user_repr = self._extract_user_representation(seq_items, seq_cates, seq_len)
                target_emb = self.model.item_emb(target_item)
                similarity = F.cosine_similarity(user_repr, target_emb, dim=1)

                # 序列统计特征
                for i in range(seq_items.size(0)):
                    valid_seq = seq_items[i, :seq_len[i]].cpu().numpy()
                    unique_items = len(np.unique(valid_seq[valid_seq != 0]))

                    sample_info = {
                        'pred_prob': probs[i].cpu().item(),
                        'true_label': 1 if labels[i].cpu().item() > 0.5 else 0,
                        'seq_len': seq_len[i].cpu().item(),
                        'target_item': target_item[i].cpu().item(),
                        'user_item_similarity': similarity[i].cpu().item(),
                        'seq_unique_items': unique_items,
                        'repeat_ratio': 1 - unique_items / max(len(valid_seq), 1),
                        'target_popularity': self.item_popularity.get(target_item[i].cpu().item(), 0),
                    }
                    samples_data.append(sample_info)

        df = pd.DataFrame(samples_data)

        # 添加预测类型标签
        df['pred_label'] = (df['pred_prob'] > 0.5).astype(int)
        df['prediction_type'] = df.apply(
            lambda row: 'TP' if row['true_label'] == 1 and row['pred_label'] == 1
            else 'TN' if row['true_label'] == 0 and row['pred_label'] == 0
            else 'FP' if row['true_label'] == 0 and row['pred_label'] == 1
            else 'FN', axis=1
        )

        return df

    def _extract_user_representation(self, seq_items, seq_cates, seq_len):
        """提取用户表示"""
        batch_size = seq_items.size(0)
        seq_len_size = seq_items.size(1)

        padding_mask = (seq_items == 0)
        item_e = self.model.item_emb(seq_items)
        cate_e = self.model.cate_emb(seq_cates)
        positions = torch.arange(seq_len_size, device=self.device).unsqueeze(0).expand(batch_size, -1)
        positions = positions * (seq_items != 0).long()
        pos_e = self.model.pos_emb(positions)

        x = item_e + cate_e + pos_e
        x = self.model.emb_dropout(x)

        causal_mask = torch.triu(torch.ones(seq_len_size, seq_len_size, device=self.device), diagonal=1).bool()

        for i in range(self.model.cfg.NUM_BLOCKS):
            x_norm = self.model.layer_norms_attn[i](x)
            attn_out, _ = self.model.attention_layers[i](
                x_norm, x_norm, x_norm,
                key_padding_mask=padding_mask,
                attn_mask=causal_mask,
                need_weights=False
            )
            x = x + attn_out

            x_norm = self.model.layer_norms_forward[i](x)
            ff_out = self.model.forward_layers[i](x_norm)
            x = x + ff_out

        x = self.model.final_norm(x)
        user_repr = x[torch.arange(batch_size), seq_len - 1]
        return user_repr

    def analyze_error_clusters(self, df, n_clusters=4):
        """【待考证】预测错误样本的特征聚类分析"""
        error_df = df[df['prediction_type'].isin(['FP', 'FN'])].copy()

        if len(error_df) < n_clusters * 10:
            print(f"警告：错误样本不足({len(error_df)})，跳过聚类")
            return None, None

        feature_cols = ['seq_len', 'user_item_similarity', 'seq_unique_items', 'target_popularity']
        X = error_df[feature_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        error_df['cluster'] = kmeans.fit_predict(X_scaled)

        # 聚类统计
        cluster_summary = error_df.groupby('cluster').agg({
            'seq_len': ['mean', 'std'],
            'target_popularity': 'mean',
            'prediction_type': lambda x: (x == 'FN').mean(),
            'pred_prob': 'mean'
        }).round(3)

        # 命名聚类
        cluster_names = {}
        for cid in range(n_clusters):
            subset = error_df[error_df['cluster'] == cid]
            fn_ratio = (subset['prediction_type'] == 'FN').mean()
            if fn_ratio > 0.6:
                name = f"高FN簇-{cid}(漏杀)"
            elif fn_ratio < 0.4:
                name = f"高FP簇-{cid}(误杀)"
            elif subset['seq_len'].mean() > 20:
                name = f"长序列簇-{cid}"
            else:
                name = f"混合簇-{cid}"
            cluster_names[cid] = name

        error_df['cluster_name'] = error_df['cluster'].map(cluster_names)

        print("\n" + "=" * 60)
        print("【误差分析 1】错误样本聚类结果")
        print("=" * 60)
        for cid, name in cluster_names.items():
            subset = error_df[error_df['cluster'] == cid]
            print(f"\n{name}:")
            print(f"  样本数: {len(subset)} ({len(subset) / len(error_df):.1%})")
            print(f"  FN比例: {(subset['prediction_type'] == 'FN').mean():.1%}")
            print(f"  平均序列长度: {subset['seq_len'].mean():.1f}")
            print(f"  平均物品热度: {subset['target_popularity'].mean():.1f}")

        return error_df, cluster_summary

    def analyze_fn_fp_distribution(self, df):
        """【待考证】False Negative / False Positive 的分布特征"""
        fp_df = df[df['prediction_type'] == 'FP']
        fn_df = df[df['prediction_type'] == 'FN']

        if len(fp_df) == 0 or len(fn_df) == 0:
            print("警告：FP或FN样本不足")
            return None

        metrics = ['seq_len', 'user_item_similarity', 'target_popularity', 'seq_unique_items']

        comparison = pd.DataFrame({
            'FP_Mean': fp_df[metrics].mean(),
            'FN_Mean': fn_df[metrics].mean(),
        })

        # KS 检验
        ks_results = {}
        for col in metrics:
            statistic, pvalue = stats.ks_2samp(fp_df[col], fn_df[col])
            ks_results[col] = {'ks_stat': statistic, 'p_value': pvalue}

        comparison['KS_Stat'] = [ks_results[m]['ks_stat'] for m in metrics]
        comparison['P_Value'] = [ks_results[m]['p_value'] for m in metrics]
        comparison['Significant'] = comparison['P_Value'] < 0.05

        print("\n" + "=" * 60)
        print("【误差分析 2】FN vs FP 分布特征对比")
        print("=" * 60)
        print(comparison.round(3))

        print("\n关键发现:")
        for feature in metrics:
            if comparison.loc[feature, 'Significant']:
                fp_m = comparison.loc[feature, 'FP_Mean']
                fn_m = comparison.loc[feature, 'FN_Mean']
                print(f"• {feature}: FP={fp_m:.2f}, FN={fn_m:.2f}, KS={comparison.loc[feature, 'KS_Stat']:.3f}")

        return comparison

    def analyze_sequence_length_impact(self, df):
        """【待考证】序列长度与预测准确性的关系"""
        df['len_bucket'] = pd.cut(df['seq_len'],
                                  bins=[0, 5, 10, 15, 20, 25, 50],
                                  labels=['1-5', '6-10', '11-15', '16-20', '21-25', '25+'])

        length_stats = df.groupby('len_bucket').agg({
            'true_label': ['count', 'mean'],
            'prediction_type': [
                lambda x: (x == 'TP').sum() / ((x == 'TP') | (x == 'FN')).sum(),  # Recall
                lambda x: (x == 'TP').sum() / ((x == 'TP') | (x == 'FP')).sum(),  # Precision
                lambda x: (x.isin(['FP', 'FN'])).mean()  # Error Rate
            ]
        })

        length_stats.columns = ['sample_count', 'actual_ctr', 'recall', 'precision', 'error_rate']
        length_stats['f1'] = 2 * (length_stats['precision'] * length_stats['recall']) / \
                             (length_stats['precision'] + length_stats['recall'] + 1e-8)

        print("\n" + "=" * 60)
        print("【误差分析 3】序列长度与预测准确性关系")
        print("=" * 60)
        print(length_stats.round(3))

        best_bucket = length_stats['f1'].idxmax()
        worst_bucket = length_stats['f1'].idxmin()
        print(f"\n最佳性能区间: {best_bucket} (F1={length_stats.loc[best_bucket, 'f1']:.3f})")
        print(f"最差性能区间: {worst_bucket} (F1={length_stats.loc[worst_bucket, 'f1']:.3f})")

        return length_stats


class PerformanceBenchmarkCPU:
    """纯 CPU 性能测试 - 解决【待考证】推理速度问题"""

    def __init__(self, model, cfg):
        self.cfg = cfg
        self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.results = {}

    def benchmark_single_sample(self, n_runs=500):
        """【待考证】单样本推理延迟（CPU 环境）"""
        print("\n[性能测试 1] 单样本推理延迟")

        dummy_input = {
            'seq_items': torch.randint(0, self.cfg.num_items, (1, self.cfg.MAX_SEQ_LEN)),
            'seq_cates': torch.randint(0, 100, (1, self.cfg.MAX_SEQ_LEN)),
            'target_item': torch.randint(0, self.cfg.num_items, (1,)),
            'seq_len': torch.tensor([self.cfg.MAX_SEQ_LEN])
        }

        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = self.model(**dummy_input)

        # 测试
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.model(**dummy_input)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

        stats = {
            'mean': np.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
        }

        print(f"  平均延迟: {stats['mean']:.2f}ms")
        print(f"  P95延迟: {stats['p95']:.2f}ms")

        self.results['single'] = stats
        return stats

    def benchmark_batch(self, batch_sizes=[1, 8, 16, 32, 64]):
        """【待考证】批量推理吞吐量"""
        print("\n[性能测试 2] 批量推理吞吐量")

        batch_results = {}
        for bs in batch_sizes:
            dummy_input = {
                'seq_items': torch.randint(0, self.cfg.num_items, (bs, self.cfg.MAX_SEQ_LEN)),
                'seq_cates': torch.randint(0, 100, (bs, self.cfg.MAX_SEQ_LEN)),
                'target_item': torch.randint(0, self.cfg.num_items, (bs,)),
                'seq_len': torch.randint(5, self.cfg.MAX_SEQ_LEN, (bs,))
            }

            with torch.no_grad():
                for _ in range(10):  # Warmup
                    _ = self.model(**dummy_input)

                latencies = []
                for _ in range(50):
                    start = time.perf_counter()
                    _ = self.model(**dummy_input)
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)

            avg_latency = np.mean(latencies)
            throughput = bs / (avg_latency / 1000)

            batch_results[bs] = {
                'latency_ms': avg_latency,
                'throughput_qps': throughput
            }

            print(f"  Batch={bs}: {avg_latency:.1f}ms, {throughput:.0f} QPS")

        self.results['batch'] = batch_results
        return batch_results

    def analyze_preprocessing_time(self):
        """【待考证】序列编码预处理耗时"""
        print("\n[性能测试 3] 序列编码预处理耗时")

        # 模拟预处理步骤
        seq_len = self.cfg.MAX_SEQ_LEN

        # 1. 序列截断/填充
        start = time.perf_counter()
        for _ in range(1000):
            seq = list(range(20))  # 模拟历史序列
            seq = seq[-seq_len:]  # 截断
            seq = [0] * (seq_len - len(seq)) + seq  # 填充
        t_truncate = (time.perf_counter() - start) / 1000 * 1000  # ms

        # 2. Item2Idx 映射
        item_map = {i: i + 2 for i in range(10000)}
        start = time.perf_counter()
        for _ in range(1000):
            mapped = [item_map.get(x, 1) for x in seq]
        t_map = (time.perf_counter() - start) / 1000 * 1000

        # 3. Tensor 转换
        start = time.perf_counter()
        for _ in range(1000):
            t = torch.tensor(seq, dtype=torch.long)
        t_tensor = (time.perf_counter() - start) / 1000 * 1000

        total = t_truncate + t_map + t_tensor

        print(f"  序列截断/填充: {t_truncate:.3f}ms ({t_truncate / total:.1%})")
        print(f"  Item2Idx 映射: {t_map:.3f}ms ({t_map / total:.1%})")
        print(f"  Tensor 转换: {t_tensor:.3f}ms ({t_tensor / total:.1%})")
        print(f"  预处理总计: {total:.3f}ms")

        self.results['preprocessing'] = {
            'truncate_ms': t_truncate,
            'map_ms': t_map,
            'tensor_ms': t_tensor,
            'total_ms': total
        }
        return self.results['preprocessing']

    def measure_memory(self):
        """测量内存占用"""
        print("\n[资源消耗] CPU 内存占用")

        process = psutil.Process()
        initial_mem = process.memory_info().rss / 1024 ** 2

        # Batch=32 推理
        dummy_input = {
            'seq_items': torch.randint(0, self.cfg.num_items, (32, self.cfg.MAX_SEQ_LEN)),
            'seq_cates': torch.randint(0, 100, (32, self.cfg.MAX_SEQ_LEN)),
            'target_item': torch.randint(0, self.cfg.num_items, (32,)),
            'seq_len': torch.randint(5, self.cfg.MAX_SEQ_LEN, (32,))
        }

        with torch.no_grad():
            _ = self.model(**dummy_input)

        current_mem = process.memory_info().rss / 1024 ** 2

        print(f"  初始内存: {initial_mem:.1f} MB")
        print(f"  Batch=32 推理后: {current_mem:.1f} MB")
        print(f"  增量: {current_mem - initial_mem:.1f} MB")

        self.results['memory'] = {
            'initial_mb': initial_mem,
            'inference_mb': current_mem,
            'model_size_mb': sum(p.numel() for p in self.model.parameters()) * 4 / 1024 ** 2
        }
        return self.results['memory']


def run_complete_analysis(model, cfg, train_loader, test_loader, save_dir):
    """
    运行完整分析（误差 + 性能）
    直接在你的 main() 函数末尾调用此函数
    """
    print("\n" + "=" * 70)
    print("SASRec 完整分析（误差分析 + 性能测试）")
    print("=" * 70)

    # ========== Part 1: 误差分析 ==========
    print("\n" + "=" * 70)
    print("第一部分：误差分析")
    print("=" * 70)

    # 计算物品流行度
    item_pop = defaultdict(int)
    for batch in train_loader:
        for item in batch['target_item'].numpy():
            item_pop[int(item)] += 1

    # 误差分析
    error_analyzer = SASRecErrorAnalyzer(model, cfg, item_popularity=item_pop)
    df = error_analyzer.extract_sample_features(test_loader, max_samples=3000)

    # 1. 聚类分析
    error_df, cluster_summary = error_analyzer.analyze_error_clusters(df)

    # 2. FN/FP 分布
    fn_fp_comp = error_analyzer.analyze_fn_fp_distribution(df)

    # 3. 序列长度影响
    length_stats = error_analyzer.analyze_sequence_length_impact(df)

    # 保存误差分析结果
    error_results = {
        'cluster_summary': cluster_summary.to_dict() if cluster_summary is not None else None,
        'fn_fp_comparison': fn_fp_comp.to_dict() if fn_fp_comp is not None else None,
        'length_impact': length_stats.to_dict()
    }

    # ========== Part 2: 性能测试 ==========
    print("\n" + "=" * 70)
    print("第二部分：性能测试（纯 CPU）")
    print("=" * 70)

    perf_benchmark = PerformanceBenchmarkCPU(model, cfg)

    # 1. 单样本延迟
    single_stats = perf_benchmark.benchmark_single_sample(n_runs=300)

    # 2. 批量吞吐量
    batch_stats = perf_benchmark.benchmark_batch(batch_sizes=[1, 8, 16, 32])

    # 3. 预处理耗时
    prep_stats = perf_benchmark.analyze_preprocessing_time()

    # 4. 内存占用
    mem_stats = perf_benchmark.measure_memory()

    # 保存性能结果
    perf_results = {
        'single_sample': single_stats,
        'batch': batch_stats,
        'preprocessing': prep_stats,
        'memory': mem_stats
    }

    # ========== 生成报告 ==========
    print("\n" + "=" * 70)
    print("生成报告")
    print("=" * 70)

    # 保存 JSON
    with open(os.path.join(save_dir, 'complete_analysis.json'), 'w') as f:
        json.dump({
            'error_analysis': error_results,
            'performance': perf_results
        }, f, indent=2)

    # 生成文本报告
    report = f"""
================================================================================
SASRec 模型分析报告（误差分析 + 性能测试）
================================================================================

一、误差分析结果
----------------
1. 错误样本聚类：识别出 {len(cluster_summary) if cluster_summary is not None else 0} 类典型错误模式
2. FN/FP 分布：见 complete_analysis.json 详细数据
3. 序列长度影响：最佳区间 {length_stats['f1'].idxmax()}, 最差区间 {length_stats['f1'].idxmin()}

二、性能测试结果（CPU 环境）
---------------------------
1. 单样本推理延迟：{single_stats['mean']:.2f}ms (P95: {single_stats['p95']:.2f}ms)
2. 批量吞吐量 (Batch=32)：{batch_stats[32]['throughput_qps']:.0f} QPS
3. 预处理耗时：{prep_stats['total_ms']:.3f}ms (截断 {prep_stats['truncate_ms']:.3f}ms + 映射 {prep_stats['map_ms']:.3f}ms + 张量 {prep_stats['tensor_ms']:.3f}ms)
4. 内存占用：模型 {mem_stats['model_size_mb']:.1f}MB，推理峰值 {mem_stats['inference_mb']:.1f}MB

三、对比基线（填入你的文档）
---------------------------
| 模型 | 单样本延迟 | Batch=32 延迟 | 吞吐量 | 硬件 |
|------|------------|---------------|--------|------|
| Logistic Regression | 0.16ms | 3.2ms | 6,250 QPS | CPU i7-10700 |
| XGBoost | 3.72ms | 118ms | 269 QPS | CPU i7-10700 |
| SASRec (CPU) | {single_stats['mean']:.2f}ms | {batch_stats[32]['latency_ms']:.1f}ms | {batch_stats[32]['throughput_qps']:.0f} QPS | CPU i7-10700 |

================================================================================
报告保存位置：{save_dir}/complete_analysis.json
================================================================================
"""

    with open(os.path.join(save_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n所有结果已保存至: {save_dir}/")

    return error_results, perf_results


# 使用说明
print("=" * 70)
print("整合版代码已准备完成！")
print("=" * 70)
print("\n在你的 main() 函数末尾添加：")
print("-" * 70)
print("""
# 导入上面的类定义

# 在 main() 最后调用：
error_results, perf_results = run_complete_analysis(
    model=model, 
    cfg=cfg, 
    train_loader=train_loader,
    test_loader=test_loader,
    save_dir=cfg.SAVE_DIR
)
""")
print("-" * 70)
print("\n这将一次性完成所有【待考证】项的分析：")
print("✓ 预测错误样本的特征聚类分析")
print("✓ False Negative / False Positive 的分布特征")
print("✓ 序列长度与预测准确性的关系")
print("✓ 单样本推理延迟（CPU 环境）")
print("✓ 批量推理吞吐量")
print("✓ 序列编码预处理耗时")
print("✓ 额外：内存占用分析")
