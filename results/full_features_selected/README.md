# 全量数据特征筛选报告
时间范围: 2014-11-18 至 2014-12-18 (共31天)
划分方式: 时间序列 8:1:1
- 训练集: 11-18 至 12-12 (25天, ~81%)
- 验证集: 12-13 至 12-15 (3天, ~10%)
- 测试集: 12-16 至 12-18 (3天, ~10%)

筛选方法:
1. IV值筛选 (threshold=0.01)
2. 相关性去冗余 (threshold=0.98)
3. LightGBM重要性排序 (Top 30)

最终特征列表:
1. ui_has_purchased
2. ui_purchase_count
3. i_purchase_ratio
4. i_browse_ratio
5. i_cart_to_buy_rate
6. ui_attention_intensity
7. ui_purchase_rate
8. i_act_total
9. ui_time_from_first_view_to_purchase
10. i_fav_to_buy_rate
11. ui_view_frequency
12. ui_views_before_purchase
13. i_act_total_14d
14. i_browse_to_buy_rate
15. ui_popularity_preference_score
16. ui_personalized_hot_score
17. i_cart_ratio
18. ui_hours_since_last_action
19. u_purchase_rate
20. u_purchase_rate_7d
21. u_purchase_rate_14d
22. i_action_trend
23. u_purchase_rate_3d
24. u_total_actions_1d
25. ui_total_actions
26. i_act_total_3d
27. u_purchase_rate_21d
28. u_total_actions_14d
29. u_purchase_rate_1d
30. u_total_actions_3d
