"""
淘宝用户行为数据预处理 - 修复中间结果保存问题
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理器 - 修复中间结果保存问题"""

    def __init__(self, config_path: str = None, chunk_size: int = 100000):
        """
        初始化预处理器

        Args:
            config_path: 配置文件路径（可选）
            chunk_size: 分块大小
        """
        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent.parent

        # 设置路径
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.processed_data_dir = self.project_root / "data" / "processed"

        # 创建输出目录
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # 分块大小
        self.chunk_size = chunk_size

        # 初始化完整统计
        self.stats = {
            'start_time': datetime.now(),
            'total_chunks': 0,
            'total_rows_processed': 0,
            'rows_removed_missing': 0,
            'rows_removed_invalid_behavior': 0,
            'rows_removed_duplicates': 0,
            'duplication_details': {
                'browse': {'original': 0, 'removed': 0, 'remaining': 0, 'strategy': '不去重'},
                'favorite': {'original': 0, 'removed': 0, 'remaining': 0, 'strategy': '不去重'},
                'cart': {'original': 0, 'removed': 0, 'remaining': 0, 'strategy': '严格去重'},
                'buy': {'original': 0, 'removed': 0, 'remaining': 0, 'strategy': '严格去重'}
            },
            'behavior_distribution': {},
            'time_range': {'min': None, 'max': None}
        }

        self.logger.info("=" * 60)
        self.logger.info("淘宝用户行为数据预处理器")
        self.logger.info("=" * 60)
        self.logger.info(f"原始数据目录: {self.raw_data_dir}")
        self.logger.info(f"处理数据目录: {self.processed_data_dir}")
        self.logger.info(f"分块大小: {chunk_size:,} 行")

    def detect_file_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(10000)
                self.logger.info(f"检测到文件编码: {encoding}")
                return encoding
            except UnicodeDecodeError:
                continue

        self.logger.warning("无法检测编码，使用默认: utf-8")
        return 'utf-8'

    def convert_data_types(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        if 'user_id' in chunk.columns:
            chunk['user_id'] = chunk['user_id'].astype(str)
        if 'item_id' in chunk.columns:
            chunk['item_id'] = chunk['item_id'].astype(str)
        if 'item_category' in chunk.columns:
            chunk['item_category'] = chunk['item_category'].astype(str)
        if 'behavior_type' in chunk.columns:
            chunk['behavior_type'] = chunk['behavior_type'].astype(str)

        return chunk

    def handle_missing_values(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 关键字段不能缺失
        key_fields = ['user_id', 'item_id', 'time']
        for field in key_fields:
            if field in chunk.columns:
                missing_count = chunk[field].isna().sum()
                if missing_count > 0:
                    chunk = chunk.dropna(subset=[field])
                    self.stats['rows_removed_missing'] += missing_count

        # 行为类型填充
        if 'behavior_type' in chunk.columns:
            mode_value = chunk['behavior_type'].mode()
            if not mode_value.empty:
                chunk['behavior_type'] = chunk['behavior_type'].fillna(mode_value.iloc[0])

        # 商品类别填充
        if 'item_category' in chunk.columns:
            chunk['item_category'] = chunk['item_category'].fillna('Unknown')

        return chunk

    def filter_invalid_behaviors(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """过滤无效行为类型"""
        if 'behavior_type' in chunk.columns:
            valid_behaviors = ['1', '2', '3', '4']
            invalid_count = (~chunk['behavior_type'].isin(valid_behaviors)).sum()

            if invalid_count > 0:
                chunk = chunk[chunk['behavior_type'].isin(valid_behaviors)]
                self.stats['rows_removed_invalid_behavior'] += invalid_count

        return chunk

    def smart_remove_duplicates(self, chunk: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        智能去重：根据行为类型采取不同策略
        """
        # 确保有必要的列
        required_cols = ['user_id', 'item_id', 'behavior_type', 'time']
        if not all(col in chunk.columns for col in required_cols):
            self.logger.warning("缺少必要列，跳过智能去重")
            return chunk, {}

        # 按行为类型分组处理
        behavior_groups = {
            '1': 'browse',  # 浏览
            '2': 'favorite',  # 收藏
            '3': 'cart',  # 加购
            '4': 'buy'  # 购买
        }

        processed_parts = []
        dup_details = {}

        for behavior_code, behavior_name in behavior_groups.items():
            # 获取该行为类型的数据
            behavior_data = chunk[chunk['behavior_type'] == behavior_code].copy()
            original_count = len(behavior_data)

            if original_count == 0:
                dup_details[behavior_name] = {
                    'original': 0,
                    'removed': 0,
                    'remaining': 0,
                    'strategy': '无数据'
                }
                continue

            # 根据行为类型应用不同去重策略
            if behavior_code == '1':  # 浏览 - 不去重
                processed_data, removed = self.process_browse_behavior(behavior_data)
                strategy = '不去重'

            elif behavior_code == '2':  # 收藏 - 不去重
                processed_data, removed = self.process_favorite_behavior(behavior_data)
                strategy = '不去重'

            elif behavior_code == '3':  # 加购 - 严格去重
                processed_data, removed = self.process_cart_behavior(behavior_data)
                strategy = '严格去重'

            elif behavior_code == '4':  # 购买 - 严格去重
                processed_data, removed = self.process_buy_behavior(behavior_data)
                strategy = '严格去重'

            else:
                processed_data, removed = behavior_data, 0
                strategy = '默认不去重'

            # 记录统计
            dup_details[behavior_name] = {
                'original': original_count,
                'removed': removed,
                'remaining': len(processed_data),
                'strategy': strategy,
                'removal_rate': removed / original_count * 100 if original_count > 0 else 0
            }

            # 更新全局统计
            self.stats['duplication_details'][behavior_name]['original'] += original_count
            self.stats['duplication_details'][behavior_name]['removed'] += removed
            self.stats['duplication_details'][behavior_name]['remaining'] += len(processed_data)
            self.stats['rows_removed_duplicates'] += removed

            processed_parts.append(processed_data)

        # 合并所有处理后的数据
        if processed_parts:
            result = pd.concat(processed_parts, ignore_index=True)
        else:
            result = pd.DataFrame(columns=chunk.columns)

        return result, dup_details

    def process_browse_behavior(self, browse_data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """处理浏览行为：不去重"""
        # 不去重，完全保留所有浏览记录
        removed = 0

        # 添加浏览序列号（同一用户对同一商品在同小时内的第几次浏览）
        browse_data['browse_key'] = (
                browse_data['user_id'].astype(str) + '_' +
                browse_data['item_id'].astype(str) + '_' +
                browse_data['time'].astype(str)
        )

        browse_data['browse_sequence'] = browse_data.groupby('browse_key').cumcount() + 1

        # 删除临时列
        browse_data = browse_data.drop(columns=['browse_key'])

        return browse_data, removed

    def process_favorite_behavior(self, fav_data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """处理收藏行为：不去重"""
        # 不去重，完全保留所有收藏记录
        removed = 0

        # 添加收藏序列号
        fav_data['favorite_key'] = (
                fav_data['user_id'].astype(str) + '_' +
                fav_data['item_id'].astype(str) + '_' +
                fav_data['time'].astype(str)
        )

        fav_data['favorite_sequence'] = fav_data.groupby('favorite_key').cumcount() + 1

        # 删除临时列
        fav_data = fav_data.drop(columns=['favorite_key'])

        return fav_data, removed

    def process_cart_behavior(self, cart_data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """处理加购行为：严格去重"""
        original_count = len(cart_data)

        # 基于用户-商品-时间（小时）去重
        result = cart_data.drop_duplicates(
            subset=['user_id', 'item_id', 'time'],
            keep='first'
        )

        removed = original_count - len(result)

        return result, removed

    def process_buy_behavior(self, buy_data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """处理购买行为：严格去重"""
        original_count = len(buy_data)

        # 基于用户-商品-时间（小时）去重
        result = buy_data.drop_duplicates(
            subset=['user_id', 'item_id', 'time'],
            keep='first'
        )

        removed = original_count - len(result)

        return result, removed

    def add_time_features(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """添加时间相关特征"""
        if 'time' in chunk.columns:
            try:
                # 解析时间
                chunk['datetime'] = pd.to_datetime(chunk['time'], format='%Y-%m-%d %H', errors='coerce')

                # 添加衍生特征
                chunk['date'] = chunk['datetime'].dt.date
                chunk['hour'] = chunk['datetime'].dt.hour
                chunk['day_of_week'] = chunk['datetime'].dt.dayofweek
                chunk['is_weekend'] = chunk['day_of_week'].isin([5, 6]).astype(int)

                # 更新全局时间范围
                chunk_min = chunk['datetime'].min()
                chunk_max = chunk['datetime'].max()

                if self.stats['time_range']['min'] is None or chunk_min < self.stats['time_range']['min']:
                    self.stats['time_range']['min'] = chunk_min
                if self.stats['time_range']['max'] is None or chunk_max > self.stats['time_range']['max']:
                    self.stats['time_range']['max'] = chunk_max

            except Exception as e:
                self.logger.warning(f"时间特征添加失败: {e}")

        return chunk

    def process_chunk(self, chunk: pd.DataFrame, chunk_num: int) -> pd.DataFrame:
        """处理单个数据块"""
        original_size = len(chunk)

        # 1. 数据类型转换
        chunk = self.convert_data_types(chunk)

        # 2. 处理缺失值
        chunk = self.handle_missing_values(chunk)

        # 3. 过滤无效行为
        chunk = self.filter_invalid_behaviors(chunk)

        # 4. 智能去重
        chunk_before_duplicates = len(chunk)
        chunk, dup_details = self.smart_remove_duplicates(chunk)
        duplicates_removed = chunk_before_duplicates - len(chunk)

        # 5. 添加时间特征
        chunk = self.add_time_features(chunk)

        # 记录处理结果
        processed_size = len(chunk)

        if chunk_num % 20 == 0:  # 每20块记录一次详细日志
            self.logger.info(
                f"块 {chunk_num}: {original_size:,} → {processed_size:,} 行 "
                f"(移除 {original_size - processed_size:,})")

        return chunk

    def save_temp_data(self, chunks: List[pd.DataFrame], batch_num: int) -> Path:
        """保存临时数据"""
        if not chunks:
            return None

        temp_file = self.processed_data_dir / f"temp_batch_{batch_num:03d}.parquet"

        try:
            temp_df = pd.concat(chunks, ignore_index=True)
            temp_df.to_parquet(temp_file, index=False)
            self.logger.info(f"✅ 保存批次 {batch_num} 到: {temp_file} ({len(temp_df):,} 行)")
            return temp_file
        except Exception as e:
            self.logger.error(f"保存临时数据失败: {e}")
            return None

    def load_all_temp_data(self) -> pd.DataFrame:
        """加载所有临时数据并合并"""
        temp_files = list(self.processed_data_dir.glob("temp_batch_*.parquet"))

        if not temp_files:
            return pd.DataFrame()

        self.logger.info(f"找到 {len(temp_files)} 个临时文件")

        all_dfs = []
        total_rows = 0

        for i, temp_file in enumerate(sorted(temp_files), 1):
            try:
                df = pd.read_parquet(temp_file)
                all_dfs.append(df)
                total_rows += len(df)
                self.logger.info(f"  加载 {temp_file.name}: {len(df):,} 行")
            except Exception as e:
                self.logger.error(f"加载临时文件失败 {temp_file}: {e}")

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            self.logger.info(f"✅ 合并完成，总计: {total_rows:,} 行")
            return final_df
        else:
            return pd.DataFrame()

    def process_file(self, filename: str) -> Dict[str, str]:
        """
        主处理方法 - 修复中间结果保存问题

        Args:
            filename: 输入文件名

        Returns:
            dict: 包含输出文件路径的字典
        """
        self.logger.info("=" * 60)
        self.logger.info(f"开始处理文件: {filename}")
        self.logger.info("=" * 60)

        try:
            # 1. 构建输入输出路径
            input_path = self.raw_data_dir / filename

            if not input_path.exists():
                raise FileNotFoundError(f"找不到文件: {input_path}")

            file_size_mb = input_path.stat().st_size / (1024 ** 2)
            self.logger.info(f"输入文件大小: {file_size_mb:.2f} MB")

            base_name = filename.replace('.csv', '_processed')
            parquet_path = self.processed_data_dir / f"{base_name}.parquet"
            csv_path = self.processed_data_dir / f"{base_name}.csv"

            self.logger.info(f"输出文件: {parquet_path}")
            self.logger.info(f"输出文件: {csv_path}")

            # 2. 检测文件编码
            encoding = self.detect_file_encoding(input_path)

            # 3. 清理之前的临时文件
            self.cleanup_temp_files()

            # 4. 分块读取和处理
            self.logger.info(f"开始分块处理，每块 {self.chunk_size:,} 行")

            chunk_reader = pd.read_csv(
                input_path,
                encoding=encoding,
                sep=',',
                chunksize=self.chunk_size,
                dtype={'time': str}
            )

            current_batch = []
            batch_num = 1
            batch_size = 0

            for chunk_num, raw_chunk in enumerate(chunk_reader, 1):
                self.stats['total_chunks'] += 1

                # 处理当前块
                processed_chunk = self.process_chunk(raw_chunk, chunk_num)

                if not processed_chunk.empty:
                    current_batch.append(processed_chunk)
                    batch_size += len(processed_chunk)
                    self.stats['total_rows_processed'] += len(processed_chunk)

                # 显示进度
                if chunk_num % 10 == 0:
                    self.logger.info(f"已处理 {chunk_num} 块，累计 {self.stats['total_rows_processed']:,} 行")

                # 每处理一定量数据保存一个批次
                if batch_size >= 500000:  # 每50万行保存一个批次
                    self.save_temp_data(current_batch, batch_num)
                    current_batch = []
                    batch_num += 1
                    batch_size = 0

            # 5. 保存最后一批数据
            if current_batch:
                self.save_temp_data(current_batch, batch_num)

            # 6. 加载并合并所有临时数据
            self.logger.info("加载并合并所有临时数据...")
            final_df = self.load_all_temp_data()

            if final_df.empty:
                raise ValueError("没有生成有效的数据")

            self.logger.info(f"最终数据大小: {len(final_df):,} 行")

            # 7. 更新行为分布统计
            if 'behavior_type' in final_df.columns:
                behavior_counts = final_df['behavior_type'].value_counts()
                self.stats['behavior_distribution'] = {
                    '1': int(behavior_counts.get('1', 0)),
                    '2': int(behavior_counts.get('2', 0)),
                    '3': int(behavior_counts.get('3', 0)),
                    '4': int(behavior_counts.get('4', 0))
                }

            # 8. 保存为Parquet格式
            self.logger.info(f"保存为Parquet格式: {parquet_path}")
            final_df.to_parquet(parquet_path, index=False)

            # 9. 保存为CSV格式
            self.logger.info(f"保存为CSV格式: {csv_path}")
            final_df.to_csv(csv_path, index=False, encoding='utf-8')

            # 10. 清理临时文件
            self.cleanup_temp_files()

            # 11. 打印完整处理摘要
            self.print_complete_summary(final_df, file_size_mb, parquet_path, csv_path)

            # 12. 返回文件路径
            return {
                'parquet': str(parquet_path),
                'csv': str(csv_path)
            }

        except Exception as e:
            self.logger.error(f"处理文件失败: {e}", exc_info=True)
            raise

    def cleanup_temp_files(self):
        """清理临时文件"""
        temp_patterns = [
            "temp_batch_*.parquet",
            "temp_chunks_*.parquet",
            "temp_*.parquet"
        ]

        for pattern in temp_patterns:
            temp_files = list(self.processed_data_dir.glob(pattern))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    self.logger.debug(f"删除临时文件: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"删除临时文件失败 {temp_file}: {e}")

    def print_complete_summary(self, df: pd.DataFrame, input_size_mb: float,
                               parquet_path: Path, csv_path: Path):
        """打印完整的处理摘要"""
        # 计算处理时间
        process_time = (datetime.now() - self.stats['start_time']).total_seconds()

        # 计算文件大小
        parquet_size = parquet_path.stat().st_size / (1024 ** 2) if parquet_path.exists() else 0
        csv_size = csv_path.stat().st_size / (1024 ** 2) if csv_path.exists() else 0

        # 构建摘要
        summary = [
            "=" * 70,
            "📊 数据处理完成报告",
            "=" * 70,
            f"⏱️  处理时间: {process_time:.1f} 秒",
            f"📁 输入文件: {input_size_mb:.1f} MB",
            "",
            "📈 处理统计:",
            f"  处理块数: {self.stats['total_chunks']}",
            f"  处理总行数: {self.stats['total_rows_processed']:,}",
            f"  最终行数: {len(df):,}",
            f"  移除缺失值: {self.stats['rows_removed_missing']:,}",
            f"  移除无效行为: {self.stats['rows_removed_invalid_behavior']:,}",
            f"  移除重复数据: {self.stats['rows_removed_duplicates']:,}",
            "",
            "🎯 智能去重效果:",
        ]

        # 智能去重详情
        for behavior, details in self.stats['duplication_details'].items():
            if details['original'] > 0:
                behavior_name = {
                    'browse': '浏览',
                    'favorite': '收藏',
                    'cart': '加购',
                    'buy': '购买'
                }.get(behavior, behavior)

                remaining = details['remaining']
                original = details['original']
                removed = details['removed']
                removal_rate = removed / original * 100 if original > 0 else 0

                summary.append(
                    f"  {behavior_name}: {remaining:,}/{original:,} 行 "
                    f"(删除 {removed:,}, {removal_rate:.1f}%) - {details['strategy']}"
                )

        summary.append("")
        summary.append("📊 文件大小对比:")
        summary.append(f"  输入CSV: {input_size_mb:10.1f} MB")
        summary.append(f"  输出Parquet: {parquet_size:8.1f} MB")
        summary.append(f"  输出CSV: {csv_size:11.1f} MB")

        if parquet_size > 0:
            compression_ratio = csv_size / parquet_size
            saved = csv_size - parquet_size
            summary.append(f"  💾 压缩效果: Parquet比CSV小{compression_ratio:.1f}倍，节省{saved:.1f} MB")

        summary.append("")
        summary.append("📈 行为类型分布:")
        behavior_names = {'1': '浏览', '2': '收藏', '3': '加购', '4': '购买'}
        total = len(df)

        for code, count in self.stats['behavior_distribution'].items():
            name = behavior_names.get(str(code), f'类型{code}')
            percent = count / total * 100 if total > 0 else 0
            summary.append(f"  {name}: {count:12,} ({percent:6.1f}%)")

        # 时间范围
        if self.stats['time_range']['min'] and self.stats['time_range']['max']:
            summary.append("")
            summary.append("⏰ 时间范围:")
            summary.append(f"  开始: {self.stats['time_range']['min']}")
            summary.append(f"  结束: {self.stats['time_range']['max']}")

        summary.append("=" * 70)

        # 输出到日志和控制台
        for line in summary:
            self.logger.info(line)
            print(line)


# 为了向后兼容
TaobaoDataPreprocessor = DataPreprocessor

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("测试数据预处理器...")

    try:
        preprocessor = DataPreprocessor()
        test_file = "user_action.csv"

        if (preprocessor.raw_data_dir / test_file).exists():
            result = preprocessor.process_file(test_file)
            print(f"\n✅ 处理完成！")
            for fmt, path in result.items():
                print(f"  {fmt.upper()}: {path}")
        else:
            print(f"❌ 测试文件不存在: {preprocessor.raw_data_dir / test_file}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()