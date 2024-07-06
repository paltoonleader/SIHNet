from setting import parse_opts 
from model import generate_model
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sets = parse_opts()
sets.gpu_id = [0]
torch.manual_seed(sets.manual_seed)
model, parameters = generate_model(sets)
print(model.module.conv_seg)
C = 2
model.module.conv_seg = nn.Sequential(
    nn.Conv3d(in_channels=2048, out_channels=C, kernel_size=1, stride=1, padding=0), 
    nn.AdaptiveAvgPool3d((1,1,1)), # 全局3D平均池化
    nn.Flatten(), # 展平输出以匹配全连接层的期望输入
    nn.Linear(C, C), # 这是一个全连接层
)
print(model.module.conv_seg)
tensor = torch.randn(1,1,56, 448, 448)
o = model.to(device)(tensor.to(device))

