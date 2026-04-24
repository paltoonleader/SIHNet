"""
Program Name: infer_from_saved_layer4_feature.py
Author: Liyang Wu
Description:
    功能:
        1) 按用户当前训练/测试调用方式构建模型
        2) 将 model.module.conv_seg 替换为分类头
        3) 读取 save_layer4_feature_case.py 保存的 layer4_feature.npy
        4) 从保存的 layer4 特征继续执行分类头推理
        5) 保存复现得到的 logits / probability / predicted class

    输入:
        1) layer4_feature.npy
        2) best_model_medicalnet_10.pth
        3) 当前模型配置 parse_opts()

    输出:
        1) reproduced_logits.npy
        2) reproduced_probabilities.npy
        3) reproduced_prediction.txt

    流程:
        1) 构建模型
        2) 替换 conv_seg 为分类头
        3) 加载训练好的权重
        4) 读取 layer4_feature.npy
        5) 直接送入 model.module.conv_seg
        6) 得到最终分类结果并保存
"""

from setting import parse_opts
from model import generate_model

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_path = r'C:\Users\Asus\Desktop\program_resnetV2\feature_case\layer4_feature.npy'
    output_dir = r'C:\Users\Asus\Desktop\program_resnetV2\feature_case_reproduced'
    weight_path = r'C:\Users\Asus\Desktop\program_resnetV2\weight'
    best_weight = os.path.join(weight_path, 'best_model_medicalnet_10.pth')

    create_dir(output_dir)

    sets = parse_opts()
    sets.input_D = 32
    sets.input_H = 512
    sets.input_W = 512
    sets.pretrain_path = r'C:\Users\Asus\Desktop\program_resnetV2\weight\resnet_10.pth'
    sets.model = 'resnet'
    sets.gpu_id = [0]
    sets.model_depth = 10

    C = 2

    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)

    model.module.conv_seg = nn.Sequential(
        nn.Conv3d(in_channels=512, out_channels=C, kernel_size=1, stride=1, padding=0),
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(C, C),
    )

    model = model.to(device)
    model.load_state_dict(torch.load(best_weight, map_location=device))
    model.eval()

    feature_np = np.load(feature_path)                      # [1, 512, D, H, W]
    feature_tensor = torch.from_numpy(feature_np).float().to(device)

    with torch.no_grad():
        logits = model.module.conv_seg(feature_tensor)      # [1, 2]
        probabilities = F.softmax(logits, dim=1)            # [1, 2]
        predicted_class = logits.argmax(dim=1).item()

    logits_np = logits.detach().cpu().numpy()
    probs_np = probabilities.detach().cpu().numpy()

    np.save(os.path.join(output_dir, 'reproduced_logits.npy'), logits_np)
    np.save(os.path.join(output_dir, 'reproduced_probabilities.npy'), probs_np)

    with open(os.path.join(output_dir, 'reproduced_prediction.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Predicted class: {predicted_class}\n')
        f.write(f'Logits: {logits_np.tolist()}\n')
        f.write(f'Probabilities: {probs_np.tolist()}\n')

    print('Feature-to-result inference completed successfully.')
    print('Feature shape:', feature_tensor.shape)
    print('Reproduced logits:', logits_np)
    print('Reproduced probabilities:', probs_np)
    print('Predicted class:', predicted_class)
    print('Output dir:', output_dir)


if __name__ == '__main__':
    main()