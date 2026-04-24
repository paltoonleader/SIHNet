
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from data import MyDataSet,img_load_files
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from setting import parse_opts 
from model import generate_model

#Data==========================================================================
train_path = r'/home2/wuliyang/ZJU_final/data/train/'
test_path = r'/home2/wuliyang/ZJU_final/data/test/'
weight_path = r'/home2/wuliyang/ZheJiang_Univsesity/prelab/weight/MedicalNet/'
latest_weight = os.path.join(weight_path, 'latest_model_medicalnet_10.pth')
best_weight = os.path.join(weight_path, 'best_model_medicalnet_10.pth')
train_dirs, test_dirs = img_load_files(train_path, test_path)
train_dataset = MyDataSet(train_path, train_dirs, Cross_Entropy=True)
test_dataset = MyDataSet(test_path, test_dirs, Cross_Entropy=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#==============================================================================

#Model=========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sets = parse_opts()
sets.input_D=32
sets.input_H=512
sets.input_W=512
sets.pretrain_path=r'C:\Users\Asus\Desktop\Work\低颅压\MedicalNet\pretrain\resnet_10_23dataset.pth'
sets.model='resnet'
sets.gpu_id = [0]
sets.model_depth = 10
C = 2
model, parameters = generate_model(sets)
torch.manual_seed(sets.manual_seed)
model = model.to(device)
model.module.conv_seg = nn.Sequential(
    nn.Conv3d(in_channels=512, out_channels=C, kernel_size=1, stride=1, padding=0), 
    nn.AdaptiveAvgPool3d((1,1,1)), # 全局3D平均池化
    nn.Flatten(), # 展平输出以匹配全连接层的期望输入
    nn.Linear(C, C), # 这是一个全连接层
)
#==============================================================================

#Hyperparameter================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5, verbose=True)
num_epochs = 50
best_loss = float('inf')
losses_train = []
losses_test = []
accuracies_train = []
accuracies_test = []
#==============================================================================

#main==========================================================================
for epoch in range(num_epochs):
    
    model.train()
    loss_train_sum, accuracy_train= 0.0, 0.0
    
    for images_train, labels_train, _ in train_loader:
        images_train = images_train.to(device)
        labels_train = labels_train.to(device)
        optimizer.zero_grad()
        outputs_train = model.to(device)(images_train)
        loss_train = criterion(outputs_train, labels_train)
        loss_train.backward()
        optimizer.step()
        outputs_train_SM = F.softmax(outputs_train, dim=1)
        loss_train_sum += loss_train.item()
        predicted_labels_train = torch.round(outputs_train_SM)
        accuracy_train += torch.sum(predicted_labels_train == labels_train).item()
        
    avg_loss_train = loss_train_sum / len(train_loader)
    avg_accuracy_train = 0.5 * accuracy_train / len(train_dataset)
    losses_train.append(avg_loss_train)
    accuracies_train.append(avg_accuracy_train)
    torch.save(model.state_dict(), latest_weight)
    
    model.eval()
    loss_test_sum, accuracy_test = 0.0, 0.0
    
    with torch.no_grad():
        for images_test, labels_test, _ in test_loader:
            images_test = images_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model.to(device)(images_test)
            loss_test = criterion(outputs_test, labels_test)
            outputs_test_SM = F.softmax(outputs_test, dim=1)
            loss_test_sum += loss_test.item()
            predicted_labels_test = torch.round(outputs_test_SM)
            accuracy_test += torch.sum(predicted_labels_test == labels_test).item()
            
    avg_loss_test = loss_test_sum / len(test_loader)
    avg_accuracy_test = 0.5 * accuracy_test / len(test_dataset)
    losses_test.append(avg_loss_test)
    accuracies_test.append(avg_accuracy_test)
    
    if avg_loss_test < best_loss:
        best_loss = avg_loss_test
        torch.save(model.state_dict(), best_weight)
        print('已保存最优模型')
    scheduler.step(avg_loss_test)
    print(f'第 [{epoch+1}/{num_epochs}] 轮',
          f'训练损失: {avg_loss_train:.4f}',
          f'训练准确率: {avg_accuracy_train:.4f}',
          f'测试损失: {avg_loss_test:.4f}',
          f'测试准确率: {avg_accuracy_test:.4f}')
#==============================================================================       