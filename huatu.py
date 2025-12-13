import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

from catboost import CatBoostRegressor, Pool
from  sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt


import shap

data = pd.read_csv("data2/allratio_Vr.csv")

X = data.drop(columns=['y'])
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = CatBoostRegressor(
    iterations=6000,
    learning_rate=0.1,
    depth=6,
    eval_metric='RMSE',
    random_seed=101,
    verbose=100
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True,plot=False)
# 保存训练好的模型
model.save_model("catboost_model_new.cbm")  # 保存为 CatBoost 格式
print("模型已保存为 'catboost_model_new.cbm'")
# 对测试集进行预测
y_pred = model.predict(X_test)
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
y_pred_train = model.predict(X_train) # 训练集

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
plt.figure(figsize=(7, 6))

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
plt.tick_params(axis='both', which='major', labelsize=16, width=3)  # labelsize调整坐标轴数字大小，width设置坐标轴线条宽度

# 显示图例，去掉边框，增加字体大小
plt.legend(
    fontsize=16,
    frameon=False
)

# plt.savefig("CAT.png", dpi=600, bbox_inches='tight')
plt.savefig('CAT_Vr.pdf', format='pdf', bbox_inches='tight')
# 显示图形
plt.show()

