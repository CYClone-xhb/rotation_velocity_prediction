import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split

import shap




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

xgb = XGBRegressor(n_estimators=600, max_depth=6)

# 训练模型
xgb.fit(X_train, y_train)

# 对测试集进行预测
y_pred = xgb.predict(X_test) # 测试集

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
y_pred_train = xgb.predict(X_train) # 训练集

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

plt.savefig("XGB.png", dpi=600, bbox_inches='tight')

# 显示图形
plt.show()


# ================= 特征重要性（Gain）条形图：升序 + 数值标签 =================
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 1. 获取特征重要性（Gain）
importance_gain = xgb.get_booster().get_score(importance_type='gain')
importance_df = (pd.DataFrame(importance_gain.items(),
                              columns=['feature', 'gain'])
                 .sort_values('gain', ascending=True))   # 升序


# 生成与特征数量匹配的颜色列表
n_feat = len(importance_df)
colors = cm.get_cmap('tab20', n_feat).colors

# 2. 画图
fig, ax = plt.subplots(figsize=(7, 6))
bars = ax.barh(importance_df['feature'],
               importance_df['gain'],
               color=colors, height=0.7)

# 3. 在柱子右侧添加数值
for bar, value in zip(bars, importance_df['gain']):
    ax.text(value + 0.01 * importance_df['gain'].max(),   # 稍微右移
            bar.get_y() + bar.get_height()/2,             # y 居中
            f'{value:.3f}',                               # 保留 3 位小数
            va='center', ha='left', fontsize=10, color='black')

# 4. 美化
ax.set_xlabel('Gain', fontsize=14, fontweight='bold', fontname='Times New Roman')
ax.set_ylabel('Features', fontsize=14, fontweight='bold', fontname='Times New Roman')
ax.set_title('Feature Importance (Gain)', fontsize=16, fontweight='bold', fontname='Times New Roman')
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()

# 5. 保存
plt.savefig("XGB_Importance_Gain_Ascending_Labeled.pdf", format='pdf', bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects

# 1. 取 Top-8
TOP_K = 8
df = (pd.DataFrame(importance_gain.items(), columns=['feat', 'gain'])
        .sort_values('gain', ascending=False)
        .head(TOP_K))

angles = np.linspace(0, 2*np.pi, TOP_K, endpoint=False)
values = df['gain'] / df['gain'].max()
labels = df['feat']

# 闭合
angles = np.concatenate([angles, [angles[0]]])
values = np.concatenate([values, [values[0]]])
labels = np.concatenate([labels, [labels[0]]])

# 2. 画图
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

# 渐变填充
cmap = plt.get_cmap('Reds')
fill_color = cmap(0.55)
ax.fill(angles, values, color=fill_color, alpha=0.35)

# 折线 + 顶点
ax.plot(angles, values, color='darkred', linewidth=2.5, marker='o', markersize=6)

# 3. 数值标签（顶点处）
for angle, val, lab in zip(angles[:-1], values[:-1], labels[:-1]):
    ax.text(angle, val + 0.08, f'{val:.2f}',
            fontsize=11, ha='center', va='center',
            color='darkred', weight='bold')

# 4. 极坐标样式
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels[:-1], fontsize=13)
ax.set_ylim(0, 1.15)                # 给标签留空间
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([f'{y:.1f}' for y in [0.2,0.4,0.6,0.8,1.0]], fontsize=11)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
ax.spines['polar'].set_visible(False)   # 去掉最外圈黑线
ax.set_title('Top-8 Feature Importance (Gain)', size=16, pad=20, weight='bold')

plt.tight_layout()
plt.savefig('XGB_Radar_Gain_Polished.pdf', dpi=600, bbox_inches='tight')
plt.show()


