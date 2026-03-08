"""
命令行接口 - 可以直接从命令行运行预处理
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import DataPreprocessor


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='淘宝用户行为数据预处理')

    parser.add_argument('--input', '-i', default='user_action.csv',
                        help='输入文件名（在data/raw目录下），默认: user_action.csv')

    parser.add_argument('--output', '-o',
                        help='输出文件名，默认: {input}_processed.parquet')

    parser.add_argument('--chunksize', '-c', type=int, default=100000,
                        help='分块大小，默认: 100000')

    parser.add_argument('--log-level', '-l', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别，默认: INFO')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    print(f"🚀 开始处理: {args.input}")

    try:
        # 初始化预处理器
        preprocessor = DataPreprocessor()

        # 处理文件
        output_path = preprocessor.process_file(
            input_file=args.input,
            output_file=args.output
        )

        if output_path:
            print(f"✅ 处理完成！输出文件: {output_path}")
            return 0
        else:
            print("❌ 处理失败！")
            return 1

    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())