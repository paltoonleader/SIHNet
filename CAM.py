from setting import parse_opts
from data import process_matrix
from model import generate_model
import os
import cv2
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from net import MedicalNetV2
import torch
import torch.nn as nn
import torch.nn.functional as F

class HookHandler:
    features = None
    gradients = None

    @staticmethod
    def forward_hook(module, input, output):
        HookHandler.features = output

    @staticmethod
    def backward_hook(module, grad_in, grad_out):
        HookHandler.gradients = grad_out[0]

#Load Model====================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = r'C:\Users\16666\Desktop\低颅压\MedicalNet\program_resnetV2\weight'
best_weight = os.path.join(weight_path, 'best_model_medicalnet_10.pth')
sets = parse_opts()
sets.input_D=32
sets.input_H=512
sets.input_W=512
sets.model='resnet'
sets.gpu_id = [0]
sets.model_depth = 10
sets.pretrain_path=r'C:\Users\16666\Desktop\work\MedicalNet\pretrain\resnet_10_23dataset.pth'
torch.manual_seed(sets.manual_seed)
model = MedicalNetV2(sets, num_classes=2)
model = model.to(device)
model.model.load_state_dict(torch.load(best_weight), strict=True)
#==============================================================================

#Load Image====================================================================
input_image = np.load(
    r'C:\Users\16666\Desktop\低颅压\data\test\4719004_4693225_0002_30_label_0.npz'
    )
image = process_matrix(input_image['image'])
image_tensor = torch.from_numpy(image)
image_tensor = image_tensor.float()
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.unsqueeze(0).to(device)
#==============================================================================

#==============================================================================
model.eval()
hook_handler = HookHandler()
target_layer = model.model.module.layer4[0].conv2
target_layer.register_forward_hook(hook_handler.forward_hook)
target_layer.register_full_backward_hook(hook_handler.backward_hook)
output_tensor = model(image_tensor.to(device))
target_class = output_tensor.argmax().item()
model.zero_grad()
output_tensor[0][target_class].backward()
weights = hook_handler.gradients.mean(dim=[2, 3], keepdim=True)
grad_cam = F.relu((weights * hook_handler.features).sum(dim=1)).squeeze().detach().cpu().numpy()
grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
#==============================================================================


depth_factor = 32 / 4
height_factor = 512 / 64
width_factor = 512 / 64
grad_cam_resized = zoom(grad_cam, (depth_factor, height_factor, width_factor))
original_image = image_tensor.squeeze().cpu().numpy()

for slice_idx in range(32):
    img_slice = original_image[slice_idx]
    grad_cam_slice = grad_cam_resized[slice_idx]
    heatmap = cm.jet(grad_cam_slice)
    heatmap = heatmap[..., :3]
    img_slice_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
    superimposed_img = heatmap * 0.4 + np.expand_dims(img_slice_normalized, axis=-1) * 0.6
    superimposed_img = superimposed_img / superimposed_img.max()
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM Visualization for slice {slice_idx}")
    plt.show()