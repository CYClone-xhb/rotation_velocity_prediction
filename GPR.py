import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入GPR相关的类
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel # 导入核函数

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv("data2/allratio_Vr.csv")

X = data.drop(columns=['y'])
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# 2. 特征缩放 (标准化) - 对GPR至关重要
# -------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("特征标准化完成。")
print("-" * 30)

# 3. 定义核函数并初始化模型
# -------------------------------------------------------------
# 定义一个组合核函数：RBF核用于建模函数关系，WhiteKernel用于建模噪声
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

# 创建GPR模型实例
# n_restarts_optimizer=10 帮助模型更好地找到最优的核函数参数，避免陷入局部最优解
gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=101)

# 4. 在缩放后的数据上训练模型
# 注意：对于大数据集，GPR的训练过程可能会非常慢
# -------------------------------------------------------------
print("开始训练GPR模型... (这可能需要一些时间)")
gpr_model.fit(X_train_scaled, y_train)
print("GPR模型训练完成。")
print("优化后的核函数: ", gpr_model.kernel_)
print("-" * 30)

# 5. 在训练集和测试集上进行预测
# -------------------------------------------------------------
# GPR可以同时返回预测均值和标准差
y_train_pred, y_train_std = gpr_model.predict(X_train_scaled, return_std=True)
y_test_pred, y_test_std = gpr_model.predict(X_test_scaled, return_std=True)


# 6. 计算并打印性能指标 (基于预测均值)
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

# 7. 查看预测的不确定性 (GPR的独特优势)
# -------------------------------------------------------------
print("测试集前5个样本的预测值和不确定性(标准差):")
for i in range(5):
    print(f"  样本 {i}: 真实值={y_test.iloc[i]:.4f}, 预测值={y_test_pred[i]:.4f} ± {y_test_std[i]:.4f}")