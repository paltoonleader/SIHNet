from setting import parse_opts 
from model import generate_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MedicalNetV2(nn.Module):
    def __init__(self, sets, num_classes=2):
        super(MedicalNetV2, self).__init__()

        self.model, _ = generate_model(sets)

        if sets.model_depth in [10, 18, 34]:
            in_channels = 512
        elif sets.model_depth in [50,101]:
            in_channels = 2048
        else:
            raise ValueError(f"Unsupported model depth: {sets.model_depth}")

        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(num_classes, num_classes)
        ]

        self.model.module.conv_seg = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model.module.conv1(x)
        x = self.model.module.bn1(x)
        x = self.model.module.relu(x)
        x = self.model.module.maxpool(x)
        x = self.model.module.layer1(x)
        x = self.model.module.layer2(x)
        x = self.model.module.layer3(x)
        x = self.model.module.layer4(x)
        x = self.model.module.conv_seg(x)
    
        return x


if __name__ == '__main__':
    
    sets = parse_opts()
    sets.gpu_id = [0]
    sets.model_depth = 10
    torch.manual_seed(sets.manual_seed)
    best_model = r'C:\Users\16666\Desktop\work\MedicalNet\best_model\best_model_medicalnet_10.pth'
    sets.pretrain_path = r'C:\Users\16666\Desktop\work\MedicalNet\pretrain\resnet_10_23dataset.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalNetV2(sets, num_classes=2, adapter=False).to(device)
    
    for param in model.model.module.parameters():
        param.requires_grad = False
        
    for name, param in model.model.module.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
    model.model.load_state_dict(torch.load(best_model), strict=True)
    
    loaded_state_dict = torch.load(best_model)
    model_state_dict = model.model.state_dict()

    # 打印成功加载的参数
    print("Successfully loaded parameters:")
    for key in loaded_state_dict:
        if key in model_state_dict and torch.equal(loaded_state_dict[key], model_state_dict[key]):
            print(key)

    # 打印未加载的参数
    print("\nParameters not loaded:")
    for key in model_state_dict:
        if key not in loaded_state_dict or not torch.equal(loaded_state_dict[key], model_state_dict[key]):
            print(key)
    
    inputs = torch.rand(1,1,32,512,512).to(device)
    onputs = model(inputs)
    print(onputs.shape)
    
    