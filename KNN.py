import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # <-- 导入标准化工具
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv("data2/allratio_Vr.csv")

X = data.drop(columns=['y'])
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# 2. 特征缩放 (标准化)
# -------------------------------------------------------------
# 创建一个标准化处理器
scaler = StandardScaler()

# 使用训练数据来拟合scaler，并转换训练数据
# 注意：scaler只能用训练集来“学习”数据的均值和方差
X_train_scaled = scaler.fit_transform(X_train)

# 使用同一个scaler来转换测试数据
X_test_scaled = scaler.transform(X_test)

print("特征标准化完成。")

# 3. 初始化模型并进行训练
# -------------------------------------------------------------
# 创建k-NN回归模型实例
# n_neighbors=5 是一个常用的起始值，表示寻找5个最近的邻居
knn_model = KNeighborsRegressor(n_neighbors=3)

# 使用“经过缩放的”训练数据对模型进行拟合
# 对于k-NN，.fit()过程非常快，因为它只是将数据存储起来
knn_model.fit(X_train_scaled, y_train)
print("k-NN回归模型训练完成。")
print("-" * 30)


# 4. 在训练集和测试集上进行预测
# -------------------------------------------------------------
# 同样，预测时也需要使用缩放后的数据
y_train_pred = knn_model.predict(X_train_scaled)
y_test_pred = knn_model.predict(X_test_scaled)


# 5. 计算并打印性能指标
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