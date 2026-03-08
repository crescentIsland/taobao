# test_fix.py
import sys
from pathlib import Path

# ⭐ 关键：添加项目根目录到Python路径
project_root = Path(__file__).parent.parent  # tests的父目录是项目根目录
sys.path.insert(0, str(project_root))

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path}")
# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessing import TaobaoDataPreprocessor

print("测试修复后的预处理器...")

try:
    # 创建实例
    processor = TaobaoDataPreprocessor()

    # 检查dtype_mapping
    print("dtype_mapping包含的列:")
    for col, dtype in processor.dtype_mapping.items():
        print(f"  {col}: {dtype}")

    # 检查是否包含'time'
    if 'time' in processor.dtype_mapping:
        print("✅ dtype_mapping包含'time'列")
    else:
        print("❌ dtype_mapping缺少'time'列")

    # 检查column_mapping
    print("\ncolumn_mapping:")
    for orig, new in processor.column_mapping.items():
        print(f"  {orig} → {new}")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback

    traceback.print_exc()