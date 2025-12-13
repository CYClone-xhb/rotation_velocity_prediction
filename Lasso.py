import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


data = pd.read_csv("data2/allratio_Vr.csv")

X = data.drop(columns=['y'])
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# 2. 初始化模型并进行训练
# -------------------------------------------------------------
# 创建Lasso回归模型实例
# alpha是正则化强度，alpha越大，惩罚越强，越多的系数会变为0
# alpha=1.0是一个常用的默认值，但最佳值通常需要通过交叉验证来寻找（例如使用LassoCV）
lasso_model = Lasso(alpha=10.0)

# 使用训练数据对模型进行拟合
lasso_model.fit(X_train, y_train)
print("Lasso回归模型训练完成。")
print("-" * 30)


# 3. 在训练集和测试集上进行预测
# -------------------------------------------------------------
y_train_pred = lasso_model.predict(X_train)
y_test_pred = lasso_model.predict(X_test)

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


# 5. 查看模型的系数 (Lasso的核心特性)
# -------------------------------------------------------------
print("Lasso模型学到的特征系数:")
# 创建一个DataFrame来清晰地显示特征名称和其对应的系数
coeffs = pd.DataFrame(
    lasso_model.coef_,
    index=X.columns,
    columns=['Coefficient']
)
print(coeffs)