"""
特征工程工具函数
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_time_decay_weights(data, current_time, half_life=7):
    """
    计算时间衰减权重

    Args:
        data: 包含时间戳的数据
        current_time: 当前时间
        half_life: 半衰期（天）

    Returns:
        衰减权重数组
    """
    # 计算时间差（天）
    time_diff = (current_time - data['behavior_time']).dt.total_seconds() / 86400

    # 指数衰减权重
    weights = np.exp(-time_diff * np.log(2) / half_life)

    return weights


def create_time_window_features(data, entity_col, current_time, windows):
    """
    创建时间窗口特征

    Args:
        data: 原始数据
        entity_col: 实体列名（如'item_id'或'user_id'）
        current_time: 当前时间
        windows: 时间窗口列表（天）

    Returns:
        DataFrame with window features
    """
    features_list = []

    for entity in data[entity_col].unique():
        entity_data = data[data[entity_col] == entity].copy()

        # 计算时间差
        entity_data['days_diff'] = (current_time - entity_data['behavior_time']).dt.days

        features = {'entity_id': entity}

        for window in windows:
            # 窗口内数据
            window_data = entity_data[entity_data['days_diff'] <= window]

            # 统计特征
            features[f'total_{window}d'] = len(window_data)

            # 行为类型统计
            for behavior in [1, 2, 3, 4]:
                count = len(window_data[window_data['behavior_type'] == behavior])
                features[f'type{behavior}_{window}d'] = count

        features_list.append(features)

    return pd.DataFrame(features_list)


def analyze_window_importance(results_df, feature_prefix='i_act_total_'):
    """
    分析不同时间窗口的重要性

    Args:
        results_df: 包含特征重要性结果
        feature_prefix: 特征前缀

    Returns:
        DataFrame with window importance analysis
    """
    # 提取时间窗口特征
    window_features = [col for col in results_df.columns if col.startswith(feature_prefix)]

    importance_data = []
    for feature in window_features:
        # 提取窗口天数
        window_days = int(feature.replace(feature_prefix, '').replace('d', ''))

        importance_data.append({
            'window_days': window_days,
            'feature_name': feature,
            'importance_mean': results_df[feature].mean() if feature in results_df.columns else 0
        })

    return pd.DataFrame(importance_data).sort_values('window_days')


def find_optimal_windows(window_importance_df, n_windows=3):
    """
    寻找最优时间窗口

    Args:
        window_importance_df: 窗口重要性分析结果
        n_windows: 要选择的最佳窗口数

    Returns:
        最优窗口列表
    """
    # 按重要性排序
    sorted_windows = window_importance_df.sort_values('importance_mean', ascending=False)

    # 选择top N
    optimal_windows = sorted_windows['window_days'].head(n_windows).tolist()

    return sorted(optimal_windows)


if __name__ == "__main__":
    # 测试工具函数
    print("Feature utilities module loaded successfully")