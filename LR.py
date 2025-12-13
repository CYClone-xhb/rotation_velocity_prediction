import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


data = pd.read_csv("data2/allratio_Vr.csv")

X = data.drop(columns=['y'])
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lr_model = LinearRegression()

# 使用训练数据对模型进行拟合
lr_model.fit(X_train, y_train)

# 3. 在训练集和测试集上进行预测
# -------------------------------------------------------------
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# 4. 计算并打印性能指标
# -------------------------------------------------------------
# -- 训练集性能 --
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("训练集性能指标:")
print(f"  R-squared (R2): {r2_train:.4f}")
print(f"  Mean Absolute Error (MAE): {mae_train:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_train:.4f}")
print("-" * 30)

# -- 测试集性能 --
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("测试集性能指标:")
print(f"  R-squared (R2): {r2_test:.4f}")
print(f"  Mean Absolute Error (MAE): {mae_test:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_test:.4f}")
print("-" * 30)