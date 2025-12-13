import shap
import torch
from IPython.core.pylabtools import figsize
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = pd.read_csv("data2/allratio_Vr.csv")

X = data.drop(columns=['y'])
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 Torch 张量
X_train_tensor = torch.from_numpy(X_train.values).float()
y_train_tensor = torch.from_numpy(y_train.values).float()
X_test_tensor = torch.from_numpy(X_test.values).float()
y_test_tensor = torch.from_numpy(y_test.values).float()

# 创建自定义数据集类
class ExcelDataset(Dataset):
    def __init__(self, features, labels):
        self.x = features
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return {'x': x, 'y': y}

# 数据加载
train_dataset = ExcelDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)

# 定义 BP 神经网络模型
class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()
        self.queue = [
            nn.Linear(10, 500), nn.BatchNorm1d(500),
            nn.Dropout(0.5),

            nn.Linear(500, 1)
        ]
        self.model = nn.Sequential(*self.queue)

    def forward(self, input):
        output = self.model(input)
        output = output.squeeze(1)
        return output

# 训练与测试设置
n_epochs = 1000
bp = BP()
if torch.cuda.is_available():
    bp = bp.cuda()
optimizer = torch.optim.AdamW(bp.parameters(), lr=0.01, weight_decay=1e-4)
MSEloss = nn.SmoothL1Loss()
if torch.cuda.is_available():
    MSEloss = MSEloss.cuda()
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# 训练函数
def train():
    y_train_list = []
    y_true_list = []  # 用于保存真实标签
    writer = SummaryWriter('lgger')
    for epoch in range(0, n_epochs):  ## for epoch in (0, 50)
        for i, value in enumerate(train_loader):
            x = value['x']
            y = value['y']
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            y_train = bp(x)
            y_train_list.extend(y_train.detach().cpu().numpy().tolist())  # 使用 extend 以确保保存为一维列表
            y_true_list.extend(y.detach().cpu().numpy().tolist())  # 使用 extend 以确保保存为一维列表
            loss = MSEloss(y, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_loader),
                    loss.item(),
                ))
            batches_done = epoch * len(train_loader) + i
            writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=batches_done)
        scheduler.step()
    torch.save(bp.state_dict(), "bp_%d.pth" % n_epochs)

    r2 = r2_score(y_true_list, y_train_list)
    mae = mean_absolute_error(y_true_list, y_train_list)
    rmse = mean_squared_error(y_true_list, y_train_list, squared=False)
    maep = np.mean(np.abs((np.array(y_true_list) - np.array(y_train_list)) / np.array(y_true_list))) * 100


    print("训练集")
    print(f"R2 Score: {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAEP): {maep}%")

    writer.close()

# 测试函数
def test():
    predict_list = []
    bp.load_state_dict(torch.load("bp_%d.pth" % n_epochs))
    bp.eval()
    for test_feature in X_test_tensor:
        test_feature = test_feature.unsqueeze(0)
        if torch.cuda.is_available():
            test_feature = test_feature.cuda()
        predict = bp(test_feature)
        predict = predict.squeeze(0)
        predict_list.append(predict.detach().cpu().numpy())

    predict_list = np.array(predict_list)
    r2 = r2_score(y_test_tensor.numpy(), predict_list)
    mae = mean_absolute_error(y_test_tensor.numpy(), predict_list)
    rmse = mean_squared_error(y_test_tensor.numpy(), predict_list, squared=False)
    maep = np.mean(np.abs((y_test_tensor.numpy() - predict_list) / y_test_tensor.numpy())) * 100


    print("测试集")
    print(f"R2 Score: {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAEP): {maep}%")



    # # 保存预测值和测试值到 Excel 文件
    # results_df = pd.DataFrame({'Observed': y_test_tensor.numpy(), 'Predicted': predict_list})
    # results_df.to_excel('prediction_results_BP_gb.xlsx', index=False)
    # print("预测值和测试值已保存到 prediction_results_BP_gb.xlsx 文件中")

# 运行训练与测试
train()
test()
# 设置全局字体为 Times New Roman
# plt.rcParams.update({'font.family': 'Times New Roman'})
#
# # 绘制散点图
# plt.figure(figsize=(8, 8))
#
# # 绘制训练集和测试集的预测结果散点图
# plt.scatter(y_train, y_pred_train, color='blue', label='Train', alpha=0.6, s=100)  # 设置散点大小
# plt.scatter(y_test, y_pred, color='red', label='Test', alpha=0.6, s=100)  # 设置散点大小
#
# # 添加1:1线（理想情况：真实值 = 预测值）
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--', linewidth=2.0, label="1:1 Line")  # 设置线条粗细
#
# # 设置图形标签和标题，增加字体大小
# plt.xlabel('True Values', fontsize=16,fontweight='bold')
# plt.ylabel('Predicted Values', fontsize=16,fontweight='bold')
# plt.title('Train and Test Predictions with 1:1 Line', fontsize=16,fontweight='bold')
#
# # 设置坐标轴刻度字体大小和加粗
# plt.tick_params(axis='both', which='major', labelsize=16, width=2)  # labelsize调整坐标轴数字大小，width设置坐标轴线条宽度
#
# # 显示图例，去掉边框，增加字体大小
# plt.legend(fontsize=16, frameon=False)
#
# plt.savefig("BP.png", dpi=600, bbox_inches='tight')
#
# # 显示图形
# plt.show()