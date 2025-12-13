# from sklearn.datasets import fetch_california_housing
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

# 加载数据集
data = pd.read_csv("data2/allratio_Vr.csv")
# print(df.head())


X = data.drop(columns=['y'])
y = data['y']
# data = pd.read_csv("data/NPP_1.csv")
#
# X = data.drop(columns=['NPP'])
# y = data['NPP']

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义随机森林回归模型
rfr = RandomForestRegressor(n_estimators=300, random_state=42)# random_state=42 确保实验结果的可重复性

# 训练模型
rfr.fit(X_train, y_train)

# 对测试集进行预测
y_pred = rfr.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
maep = np.mean(np.abs((y_test.values.ravel() - y_pred) / y_test.values.ravel())) * 100

print("测试集")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAEP): {maep}%")


# 对训练集进行预测
y_pred_train = rfr.predict(X_train) # 训练集

r2 = r2_score(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train, squared=False)
maep = np.mean(np.abs((y_train.values.ravel() - y_pred_train) / y_train.values.ravel())) * 100

print("训练集")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAEP): {maep}%")

# 设置全局字体为 Times New Roman
plt.rcParams.update({'font.family': 'Times New Roman'})

# 绘制散点图
plt.figure(figsize=(6, 6))

# 绘制训练集和测试集的预测结果散点图
plt.scatter(y_train, y_pred_train, color='blue', label='Train', alpha=0.6, s=100)  # 设置散点大小
plt.scatter(y_test, y_pred, color='red', label='Test', alpha=0.6, s=100)  # 设置散点大小

# 添加1:1线（理想情况：真实值 = 预测值）
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--', linewidth=2.0, label="1:1 Line")  # 设置线条粗细

# 设置图形标签和标题，增加字体大小
plt.xlabel('True Values', fontsize=16,fontweight='bold')
plt.ylabel('Predicted Values', fontsize=16,fontweight='bold')
plt.title('', fontsize=16,fontweight='bold')

# 设置坐标轴刻度字体大小和加粗
plt.tick_params(axis='both', which='major', labelsize=16, width=2)  # labelsize调整坐标轴数字大小，width设置坐标轴线条宽度

# 显示图例，去掉边框，增加字体大小
plt.legend(fontsize=16, frameon=False)

plt.savefig("RF.png", dpi=600, bbox_inches='tight')

# 显示图形
plt.show()


# 输出特征重要性
importances = rfr.feature_importances_
indices = np.argsort(importances)[::-1]

# 打印每个特征的重要性
for f in range(X.shape[1]):
    print(f"特征 {f + 1}: {X.columns[indices[f]]} ({importances[indices[f]]})")

# 可视化特征重要性
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])

plt.gcf().set_size_inches(7, 6)
plt.savefig('RFFI', dpi=600, bbox_inches='tight')
plt.show()
#
#
# # 绘制预测值与实际值的散点图
# plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Random Forest Regression: Actual vs Predicted")
# plt.show()

# # 保存预测值和测试值到Excel文件
# results_df = pd.DataFrame({'Observed': y_test.values.ravel(), 'Predicted': y_pred})
# results_df.to_excel('prediction_results_RF.xlsx', index=False)
# print("预测值和测试值已保存到 prediction_results_RF.xlsx 文件中")
#
# # SHAP 分析
# explainer = shap.TreeExplainer(rfr)
# shap_values = explainer.shap_values(X)
#
# # 设置字体
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.rcParams['font.size'] = 25
#
# # summary_plot (bar)
# shap.summary_plot(shap_values, X, plot_type="bar", show=False)
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('summary_plot_bar.png', dpi=600, bbox_inches='tight')
# plt.close()
#
# # summary_plot (dot)
# shap.summary_plot(shap_values, X, plot_type="dot", show=False)
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('summary_plot_dot.png', dpi=600, bbox_inches='tight')
# plt.close()
#
# # force_plot
# fig = shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], feature_names=X.columns, matplotlib=True, show=False)
# # shap.save_html('force_plot.html', shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], feature_names=X.columns))
# # 保存为 PNG 图像
# fig.savefig('force_plot.png', dpi=600, bbox_inches='tight')
# plt.close(fig)
#
# # waterfall plot
# shap_waterfall = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X.iloc[0, :], feature_names=X.columns)
# shap.plots.waterfall(shap_waterfall, show=False)
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('waterfall.png', dpi=600, bbox_inches='tight')
# plt.close()
#
# # heatmap plot
# shap_heatmap = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X, feature_names=X.columns)
# shap.plots.heatmap(shap_heatmap, show=False)
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('heatmap.png', dpi=600, bbox_inches='tight')
# plt.close()
#
# # dependence_plot
# shap.dependence_plot('ρ', shap_values, X, interaction_index='z', show=False)
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('dependence_plot.png', dpi=600, bbox_inches='tight')
# plt.close()
