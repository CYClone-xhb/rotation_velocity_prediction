import numpy as np
import pandas as pd
import shap
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# 加载数据集
data = pd.read_csv("data2/allratio_Vr.csv")
# print(df.head())


X = data.drop(columns=['y'])
y = data['y']
# y = np.ravel(y)
#标准化
std = StandardScaler()
X_std  =std.fit_transform(X)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size= 0.2)

# 定义参数网格
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.1, 1],
    'kernel': ['rbf', 'linear']
}

# 使用GridSearchCV进行参数调优
svm = SVR()
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


# 获取最佳参数和最佳模型
best_params = grid_search.best_params_
print(f"最佳参数: {best_params}")


# SVM建模
# svm_regression = SVR()
svm_regression = grid_search.best_estimator_
svm_regression.fit(X_train, y_train)
y_pred = svm_regression.predict(X_test)

# 模型效果
# svm_regression.score(X_test, y_test)
# print(svm_regression.score(X_test, y_test))

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
y_pred_train = svm_regression.predict(X_train) # 训练集

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

plt.savefig("SVM.png", dpi=600, bbox_inches='tight')

# 显示图形
plt.show()

# plt.figure((figsize(15,6)))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.title("observed vs predicted values")
# plt.xlabel("observed")
# plt.ylabel("predicted")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
# plt.show()
#
# # 保存预测值和测试值到Excel文件
# results_df = pd.DataFrame({'Observed': y_test, 'Predicted': y_pred})
# results_df.to_excel('prediction_results_SVM.xlsx', index=False)
# print("预测值和测试值已保存到 prediction_results_SVM.xlsx 文件中")
#
# # 使用 SHAP 分析 SVM 模型
# explainer = shap.KernelExplainer(svm_regression.predict, X_train[:100])  # 使用部分训练数据作为背景数据
# shap_values = explainer.shap_values(X_test[:50])  # 对部分测试数据计算 SHAP 值
#
# # 设置字体
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.rcParams['font.size'] = 25
#
# # summary_plot (bar)
# shap.summary_plot(shap_values, X_test[:50], plot_type="bar", show=False)  # 使用与 shap_values 一致的数据集
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('summary_plot_bar.png', dpi=600, bbox_inches='tight')
# plt.close()
#
# # summary_plot (dot)
# shap.summary_plot(shap_values, X_test[:50], plot_type="dot", show=False)  # 使用与 shap_values 一致的数据集
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('summary_plot_dot.png', dpi=600, bbox_inches='tight')
# plt.close()
#
# # force_plot
# fig = shap.force_plot(explainer.expected_value, shap_values[0, :], X_test[0, :], feature_names=X.columns, matplotlib=True, show=False)  # 使用 X_test
# # 保存为 PNG 图像
# fig.savefig('force_plot.png', dpi=600, bbox_inches='tight')
# plt.close(fig)
#
# # waterfall plot
# shap_waterfall = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test[0, :], feature_names=X.columns)  # 使用 X_test
# shap.plots.waterfall(shap_waterfall, show=False)
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('waterfall.png', dpi=600, bbox_inches='tight')
# plt.close()
#
# # heatmap plot
# shap_heatmap = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_test[:50], feature_names=X.columns)  # 使用 X_test
# shap.plots.heatmap(shap_heatmap, show=False)
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('heatmap.png', dpi=600, bbox_inches='tight')
# plt.close()
#
# # dependence_plot
# shap.dependence_plot('ρ', shap_values, X_test[:50], interaction_index='z', show=False)  # 使用 X_test
# plt.gcf().set_size_inches(7, 6)
# plt.savefig('dependence_plot.png', dpi=600, bbox_inches='tight')
# plt.close()



# # summary_plot
# shap.summary_plot(shap_values, X_test[:50], feature_names=X.columns, plot_type="bar")   # summary_plot 条状
# shap.summary_plot(shap_values, X_test[:50], feature_names=X.columns, plot_type="dot")   # summary_plot 点状
#
# # force_plot
# shap.force_plot(explainer.expected_value, shap_values[0], X_test[0], feature_names=X.columns, matplotlib=True)
#
# # waterfall plot
# shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test[0], feature_names=X.columns))
#
# # heatmap plot
# shap.plots.heatmap(shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_test, feature_names=X.columns))
#
# # dependence_plot
# shap.dependence_plot('x1', shap_values, X_test, interaction_index='x2')