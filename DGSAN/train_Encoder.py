import sys
sys.path.append('./')
import time
import torch.nn as nn
from model.observer import Runtime_Observer
from dataloader.dataset_yu import *
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from model.Encoder import Encoder_Tiny
import os
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
import torch
import math

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-2):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

# 使用示例：
csv_path = ''  # 替换为你的 CSV 文件的路径
data_dir = ''  # 替换为你的 .npy 文件所在的目录路径
text_csv_path = ''
csv_data = pd.read_csv(csv_path)
text_data = pd.read_csv(text_csv_path)

# 创建 KFold 对象
# kf = KFold(n_splits=10, shuffle=True, random_state=42)
# 将 subject_ids 转换为列表
subject_ids = csv_data['Subject ID'].unique()
num_epochs = 100
batch_size = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练和验证集划分
train_ids, val_ids = train_test_split(
    subject_ids, test_size=0.2, random_state=42)  # 80% 训练集，20% 验证集

# 划分训练集和验证集
train_data = csv_data[csv_data['Subject ID'].isin(train_ids)]
val_data = csv_data[csv_data['Subject ID'].isin(val_ids)]

# 创建数据集
train_dataset = LN_text_Dataset(
    train_data, data_dir, text_data, normalize=True, augment_minority_class=False)
val_dataset = LN_text_Dataset(val_data, data_dir, text_data, normalize=True)
# print(train_dataset.num_cat)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 生成模型
model = Encoder_Tiny(num_classes=128, pretrain_path=None)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), num_epochs,
                                   warmup=True, warmup_epochs=1)

# 初始化模型、损失函数和优化器
best_val_loss = float('inf')  # 最优验证集损失
train_losses, val_losses = [], []  # 记录损失
val_accuracies, val_aucs, val_f1_scores = [], [], []  # 记录验证指标
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
if not os.path.exists(f"debug"):
    os.makedirs(f"debug")
observer = Runtime_Observer(
    log_dir=f"debug", device=device, name="debug", seed=42)
model.to(device)
num_params = 0
for p in model.parameters():
    if p.requires_grad:
        num_params += p.numel()
print("\n===============================================\n")
print("model parameters: " + str(num_params))
print("\n===============================================\n")
# 训练过程
def train_model(model, train_loader, val_loader, device, optimizer, criterion, lr_scheduler, num_epochs=10, scheduler=None):
    train_losses = []
    val_losses = []
    auc_scores = []
    accuracies = []  # 用于记录准确率
    epoch_steps = len(train_loader)
    best_val_loss = float('inf')  # 最优验证集损失
    start_time = time.time()
    observer.log("start training\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        observer.reset()

        # 训练阶段
        with tqdm(total=epoch_steps, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch in train_loader:
                inputs1, inputs2, labels, _ = batch['T0_image'], batch['T1_image'], batch['label'], batch['table_info']
                inputs1 = inputs1.unsqueeze(1).to(device)  # 添加通道维度
                inputs2 = inputs2.unsqueeze(1).to(device)  # 添加通道维度
                # 确保标签为 LongTensor
                labels = labels.to(device, dtype=torch.long)
                optimizer.zero_grad()
                
                inputs_group = [inputs1, inputs2]
                inputs = random.choice(inputs_group)
                # outputs = model(inputs)[0]
                x_fu, x_g, x_l = model(inputs)
                loss_fu = criterion(x_fu, labels)
                # loss_g = criterion(x_g, labels)
                # loss_l = criterion(x_l, labels)
                # loss = (loss_fu + loss_g + loss_l)/3
                loss_fu.backward()
                optimizer.step()
                lr_scheduler.step()

                if scheduler:
                    scheduler.step()
                running_loss += loss_fu.item()*inputs1.size(0)

                # 计算准确率
                _, preds = torch.max(x_fu, 1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                pbar.set_postfix({'Loss': running_loss / (pbar.n + 1),
                                 'Accuracy': correct_predictions / total_samples})
                pbar.update()

        train_losses = running_loss / (len(train_loader.dataset))
        accuracies.append(correct_predictions / total_samples)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        observer.log(f"Loss: {train_losses:.4f}\n")

        with torch.no_grad():
            for batch in val_loader:
                inputs1, inputs2, labels, _ = batch['T0_image'], batch[
                    'T1_image'], batch['label'], batch['table_info']
                inputs1 = inputs1.unsqueeze(1).to(device)  # 添加通道维度
                inputs2 = inputs2.unsqueeze(1).to(device)
                # 确保标签为 LongTensor
                labels = labels.to(device, dtype=torch.long)
                inputs_group = [inputs1, inputs2]
                inputs = random.choice(inputs_group)
                # outputs = model(inputs)[0]
                x_fu, x_g, x_l = model(inputs)
                loss_fu = criterion(x_fu, labels)
                # loss_g = criterion(x_g, labels)
                # loss_l = criterion(x_l, labels)
                # loss = (loss_fu + loss_g + loss_l)/3
                val_loss += loss_fu.item()*inputs1.size(0)

                # 计算准确率
                probabilities = torch.softmax(x_fu, dim=1)
                _, predictions = torch.max(x_fu, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]
                observer.update(predictions, labels, confidence_scores)

            val_losses = val_loss / len(val_loader)
            observer.log(f"Test Loss: {val_losses:.4f}\n")

        observer.record_loss(epoch, train_losses, val_losses)
        if observer.excute(epoch):
            print("Early stopping")
            break

            # 如果是最优模型，保存权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
        final_model_wts = model.state_dict()
    torch.save(best_model_wts, '')
    # torch.save(final_model_wts, 'outputs/Encoder_final_Base.pth')
    end_time = time.time()
    observer.log(f"\nRunning time: {end_time - start_time:.2f} second\n")
    observer.finish()
    return train_losses, val_losses, auc_scores, accuracies


# 训练模型
train_losses, val_losses, auc_scores, _ = train_model(
    model, train_loader, val_loader, device, optimizer, criterion, lr_scheduler, num_epochs=200)
