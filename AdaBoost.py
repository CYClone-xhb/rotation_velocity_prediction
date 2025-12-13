import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split



# 加载数据集
data = pd.read_csv("data2/allratio_Vr.csv")
# print(df.head())


X = data.drop(columns=['y'])
y = data['y']

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 输出划分的测试集
print("Test Set (X_test):")
print(X_test)
print("Test Set (y_test):")
print(y_test)

# mm1 = MinMaxScaler()   # 特征进行归一化
# X_train_m = mm1.fit_transform(X_train)
# mm2 = MinMaxScaler()     # 标签进行归一化
# y_train_m = mm2.fit_transform(y_train)

ada = AdaBoostRegressor(n_estimators=25, random_state=42)

# 训练模型
ada.fit(X_train, y_train)

# 对测试集进行预测
y_pred = ada.predict(X_test)

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
y_pred_train = ada.predict(X_train) # 训练集

r2 = r2_score(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train, squared=False)
maep = np.mean(np.abs((y_train.values.ravel() - y_pred_train) / y_train.values.ravel())) * 100

print("训练集")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAEP): {maep}%")

# # 绘制预测值与实际值的散点图
# plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("AdaBoost Regression: Actual vs Predicted")
# plt.show()

# # 保存预测值和测试值到Excel文件
# results_df = pd.DataFrame({'Observed': y_test, 'Predicted': y_pred})
# results_df.to_excel('prediction_results_AdaBoost.xlsx', index=False)
# print("预测值和测试值已保存到 prediction_results_AdaBoost.xlsx 文件中")

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

plt.savefig("Ada.png", dpi=600, bbox_inches='tight')

# 显示图形
plt.show()