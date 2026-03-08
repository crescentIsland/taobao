#!/usr/bin/env python
"""
淘宝用户行为数据预处理主脚本
基于智能去重策略（按行为类型不同处理）
用法: python scripts/run_preprocessing.py
"""

import sys
from pathlib import Path
import logging
import json
import pandas as pd  # 移到文件开头
from datetime import datetime  # 移到文件开头

# ⭐ 关键修复：设置正确的项目根目录
PROJECT_ROOT = Path(__file__).parent.parent  # scripts的父目录是项目根目录
sys.path.insert(0, str(PROJECT_ROOT))

print(f"项目根目录: {PROJECT_ROOT}")


def setup_logging():
    """配置日志"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "preprocess.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def print_file_info(file_path: Path):
    """打印文件信息"""
    if file_path.exists():
        size_mb = file_path.stat().st_size / 1024 ** 2
        print(f"   ✅ 存在 ({size_mb:.2f} MB)")
        return True
    else:
        print("   ❌ 不存在")
        return False


def create_default_config(config_file: Path):
    """创建默认配置文件"""
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("""# 项目路径配置
paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  external_data: "data/external"

# 数据预处理配置
data_preprocessing:
  chunksize: 100000

  # 智能去重策略
  deduplication_strategy:
    browse: "no_deduplicate"      # 浏览行为：不去重，只标记序列
    favorite: "mark_duplicates"   # 收藏行为：去重但记录次数
    cart: "strict_deduplicate"    # 加购行为：严格去重
    buy: "strict_deduplicate"     # 购买行为：严格去重

  # 行为类型映射
  behavior_type_mapping:
    "1": "browse"
    "2": "favorite"
    "3": "cart"
    "4": "buy"

  # 缺失值处理
  missing_values:
    user_id: "drop"
    item_id: "drop"
    behavior_type: "mode"
    item_category: "Unknown"
    time: "drop"
""")


def print_processing_summary(stats: dict, final_df=None):
    """打印处理摘要"""
    print("\n" + "=" * 60)
    print("📊 数据处理摘要")
    print("=" * 60)

    print(f"总数据块数: {stats.get('total_chunks', 0)}")
    print(f"总处理行数: {stats.get('total_rows_processed', 0):,}")

    if final_df is not None:
        print(f"最终数据行数: {len(final_df):,}")

    print(f"删除缺失值: {stats.get('rows_removed_missing', 0):,}")
    print(f"删除无效行为: {stats.get('rows_removed_invalid_behavior', 0):,}")

    # 智能去重效果
    print("\n🎯 智能去重效果:")
    dup_details = stats.get('duplication_details', {})
    for behavior, details in dup_details.items():
        original = details.get('original', 0)
        removed = details.get('removed', 0)

        if original > 0:
            behavior_names = {
                'browse': '浏览',
                'favorite': '收藏',
                'cart': '加购',
                'buy': '购买'
            }
            behavior_name = behavior_names.get(behavior, behavior)

            remaining = original - removed
            removal_rate = removed / original * 100 if original > 0 else 0

            print(f"  {behavior_name}: {remaining:,}/{original:,} 行 "
                  f"(删除 {removed:,}, {removal_rate:.1f}%) - {details.get('strategy', 'N/A')}")

    # 行为分布（如果有最终数据）
    if final_df is not None and 'behavior_type' in final_df.columns:
        behavior_counts = final_df['behavior_type'].value_counts()
        behavior_names = {
            '1': '浏览 (pv)',
            '2': '收藏 (fav)',
            '3': '加购 (cart)',
            '4': '购买 (buy)'
        }

        print("\n📈 行为类型分布:")
        total = len(final_df)
        for behavior_code, count in behavior_counts.items():
            behavior_name = behavior_names.get(str(behavior_code), f'类型{behavior_code}')
            percentage = count / total * 100
            print(f"  {behavior_name}: {count:,} ({percentage:.1f}%)")


def save_processing_report(stats: dict, output_path: Path, final_df=None):
    """保存处理报告"""
    report = {
        'summary': {
            'processed_date': str(datetime.now()),
            'input_file': 'user_action.csv',
            'output_file': str(output_path),
            'total_chunks': stats.get('total_chunks', 0),
            'total_rows_processed': stats.get('total_rows_processed', 0),
            'rows_removed_missing': stats.get('rows_removed_missing', 0),
            'rows_removed_invalid_behavior': stats.get('rows_removed_invalid_behavior', 0),
            'rows_removed_duplicates': stats.get('rows_removed_duplicates', 0)
        },
        'duplication_analysis': stats.get('duplication_details', {}),
        'processing_stats': stats
    }

    # 如果有最终数据，添加更多信息
    if final_df is not None:
        report['summary']['processed_rows'] = len(final_df)
        if 'user_id' in final_df.columns:
            report['summary']['unique_users'] = int(final_df['user_id'].nunique())
        if 'item_id' in final_df.columns:
            report['summary']['unique_items'] = int(final_df['item_id'].nunique())

        # 时间范围
        if 'datetime' in final_df.columns:
            report['summary']['time_range'] = {
                'start': str(final_df['datetime'].min()),
                'end': str(final_df['datetime'].max())
            }

    # 保存报告
    report_file = output_path.with_suffix('.report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_file


def main():
    """主函数 - 简洁版"""
    # 设置日志
    logger = setup_logging()

    print("=" * 60)
    print("淘宝用户行为数据预处理")
    print("=" * 60)

    try:
        # 1. 检查数据文件
        data_file = PROJECT_ROOT / "data" / "raw" / "user_action.csv"
        print(f"\n📂 检查数据文件...")

        if not data_file.exists():
            print(f"❌ 找不到文件: {data_file}")
            print(f"请将 user_action.csv 放在: {data_file.parent}")
            return 1

        size_mb = data_file.stat().st_size / 1024 ** 2
        print(f"  文件: {data_file.name}")
        print(f"  大小: {size_mb:.1f} MB")

        # 2. 检查配置文件
        config_file = PROJECT_ROOT / "config" / "paths.yaml"
        print(f"\n📄 检查配置文件...")

        if not config_file.exists():
            print(f"⚠️  配置文件不存在，创建默认配置...")
            config_file.parent.mkdir(exist_ok=True)
            config_file.write_text("""
paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
""")
            print(f"✅ 已创建默认配置")

        # 3. 导入预处理器
        print(f"\n🚀 导入预处理模块...")
        try:
            from src.data.preprocessing import DataPreprocessor
            print("✅ 导入成功")
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            print("请确保:")
            print("1. src/data/preprocessing.py 文件存在")
            print("2. 文件中定义了 DataPreprocessor 类")
            print("3. 已安装依赖: pip install pandas numpy pyarrow chardet PyYAML")
            return 1

        # 4. 初始化预处理器
        print(f"\n🔧 初始化预处理器...")
        try:
            preprocessor = DataPreprocessor(
                config_path=str(config_file),
                chunk_size=100000
            )
            print("✅ 初始化成功")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            print("可能是配置文件中路径键名错误")
            print("请检查 config/paths.yaml 是否包含: data_raw 和 data_processed")
            return 1

        # 5. 开始处理
        print(f"\n🔄 开始处理数据...")
        print(f"   分块大小: {preprocessor.chunk_size:,} 行/块")
        print("   处理中，请稍候...")
        print("-" * 60)

        try:
            # 调用预处理
            output_paths = preprocessor.process_file("user_action.csv")

            if not output_paths:
                print("❌ 处理失败：未返回输出路径")
                return 1

        except Exception as e:
            print(f"❌ 处理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # 6. 显示结果
        print(f"\n" + "=" * 60)
        print("✅ 处理完成！")
        print("=" * 60)

        # 检查输出文件
        files_created = []
        for fmt, path in output_paths.items():
            file_path = Path(path)
            if file_path.exists():
                size_mb = file_path.stat().st_size / 1024 ** 2
                files_created.append((fmt.upper(), path, size_mb))

        if not files_created:
            print("❌ 未生成任何输出文件")
            return 1

        print("\n📁 生成的文件:")
        for fmt, path, size_mb in files_created:
            print(f"  • {fmt}: {path}")
            print(f"     大小: {size_mb:.1f} MB")

        # 7. 文件大小对比
        csv_path = None
        parquet_path = None

        for fmt, path, size_mb in files_created:
            if fmt == 'CSV':
                csv_path = path
                csv_size = size_mb
            elif fmt == 'PARQUET':
                parquet_path = path
                parquet_size = size_mb

        if csv_path and parquet_path:
            print(f"\n📊 压缩效果:")
            print(f"  CSV:     {csv_size:.1f} MB")
            print(f"  Parquet: {parquet_size:.1f} MB")
            if parquet_size > 0:
                ratio = csv_size / parquet_size
                saved = csv_size - parquet_size
                print(f"  Parquet 比 CSV 小 {ratio:.1f} 倍")
                print(f"  节省空间: {saved:.1f} MB")

        # 8. 使用提示
        print(f"\n💡 使用提示:")
        print(f"  1. 查看数据: 用 Excel 或文本编辑器打开 CSV 文件")
        print(f"  2. 分析数据: 用 pandas.read_parquet() 读取 Parquet 文件")

        # 9. 尝试读取数据展示基本信息
        if parquet_path:
            try:
                print(f"\n📊 读取处理后的数据...")
                df = pd.read_parquet(parquet_path)

                print(f"  总行数: {len(df):,}")
                print(f"  总列数: {len(df.columns)}")

                if 'behavior_type' in df.columns:
                    print(f"\n🎯 行为类型分布:")
                    behavior_counts = df['behavior_type'].value_counts()
                    behavior_names = {
                        '1': '浏览', '2': '收藏', '3': '加购', '4': '购买'
                    }

                    for code, count in behavior_counts.items():
                        name = behavior_names.get(str(code), f'类型{code}')
                        percent = count / len(df) * 100
                        print(f"    {name}: {count:,} ({percent:.1f}%)")

                if 'datetime' in df.columns:
                    print(f"\n⏰ 时间范围:")
                    print(f"    开始: {df['datetime'].min()}")
                    print(f"    结束: {df['datetime'].max()}")

            except Exception as e:
                print(f"⚠️  无法读取输出文件: {e}")

        print(f"\n" + "=" * 60)
        print("🎉 所有任务完成！")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print(f"\n\n⏹️  用户中断处理")
        return 1

    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())