import torch
import torch.nn as nn
import timm
from PIL import Image
import os
import numpy as np
from torch import nn,optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

def img_load_files(train_path, test_path):
    train_dirs = [file for file in os.listdir(train_path) if file.endswith('.npz')]
    test_dirs = [file for file in os.listdir(test_path) if file.endswith('.npz')]
    
    return train_dirs, test_dirs

def process_matrix(matrix):
    d, h, w = matrix.shape

    if d > 32:
        start_slice = (d - 32) // 2
        end_slice = start_slice + 32
        processed_matrix = matrix[start_slice:end_slice, :, :]
    else:
        pad_before = (32 - d) // 2
        pad_after = 32 - d - pad_before
        processed_matrix = np.pad(matrix, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')

    return processed_matrix

def Cross_Entropy_label(label):
    if label == 1:
        label = torch.tensor([0, 1])
    elif label == 0:
        label = torch.tensor([1, 0])
    return label

class MyDataSet(Dataset):
    def __init__(self, path, filename, Cross_Entropy=False):
        self.filename = filename
        self.path = path
        self.Cross_Entropy = Cross_Entropy

    def __getitem__(self, index):
        filename = os.path.join(self.path, self.filename[index])
        file = np.load(filename)
        image = file['image']
        label = file['label']
        
        processed_image = process_matrix(image)
        image_tensor = torch.from_numpy(processed_image)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.unsqueeze(0)
        label_tensor = torch.tensor(label)  # Assuming label is already in the right format
        if self.Cross_Entropy:
            label_tensor = Cross_Entropy_label(label_tensor)
        else:
            label_tensor = label_tensor.unsqueeze(0)
        label_tensor = label_tensor.float()
        return image_tensor, label_tensor, filename

    def __len__(self):
        return len(self.filename)

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = r'C:\Users\16666\Desktop\ZheJiang_Univsesity\train'
    test_path = r'C:\Users\16666\Desktop\ZheJiang_Univsesity\test'
    train_dirs, test_dirs = img_load_files(train_path, test_path)
    train_dataset = MyDataSet(train_path, train_dirs, Cross_Entropy=False)
    test_dataset = MyDataSet(test_path, test_dirs, Cross_Entropy=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for images, labels, filenames in train_loader:
        print('images:',images.shape)
        print('labels:',labels)
        print('filenames:',filenames)
    for images, labels, filenames in test_loader:
        print('images:',images.shape)
        print('labels:',labels)
        print('filenames:',filenames)