import pandas as pd
import os

RESULTS_DIR = r'E:\PycharmeProjects\taobao_user_behavior_analysis\results\full_features_selected'

# 定义泄露特征
leakage_features = ['ui_has_purchased', 'ui_purchase_count']

print("开始清洗数据，删除泄露特征...\n")

for split in ['train', 'val', 'test']:
    # 1. 读取文件
    input_path = os.path.join(RESULTS_DIR, f'{split}_selected_30.parquet')
    df = pd.read_parquet(input_path)

    original_cols = set(df.columns)

    # 2. 删除泄露特征（errors='ignore' 防止报错）
    df_clean = df.drop(columns=leakage_features, errors='ignore')

    deleted = original_cols - set(df_clean.columns)
    remaining_features = [c for c in df_clean.columns if c not in ['user_id', 'item_id', 'label']]

    # 3. 保存为新文件（加_clean后缀以示区分）
    output_path = os.path.join(RESULTS_DIR, f'{split}_selected_clean.parquet')
    df_clean.to_parquet(output_path)

    # 同时保存CSV版本（方便查看）
    csv_path = os.path.join(RESULTS_DIR, f'{split}_selected_clean.csv')
    df_clean.to_csv(csv_path, index=False)

    print(f"[{split}] 处理完成:")
    print(f"  原特征数: {len(original_cols) - 3}个")  # 减去user_id, item_id, label
    print(f"  删除特征: {list(deleted)}")
    print(f"  剩余特征: {len(remaining_features)}个")
    print(f"  样本数: {len(df_clean):,}, 正样本率: {df_clean['label'].mean():.2%}")
    print(f"  保存路径: {output_path}\n")

# 4. 更新特征列表文档
feature_list_path = os.path.join(RESULTS_DIR, 'selected_features_clean.txt')
with open(feature_list_path, 'w', encoding='utf-8') as f:
    f.write("# 清洗后的特征列表（已删除泄露特征）\n\n")
    for i, feat in enumerate(remaining_features, 1):
        f.write(f"{i}. {feat}\n")

print(f"✓ 全部完成！新的特征文件已保存")
print(f"✓ 特征列表已更新: {feature_list_path}")