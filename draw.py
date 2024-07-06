from torchvision import transforms
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from data import MyDataSet, img_load_files
from tool import plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.nn.functional as F
from setting import parse_opts
from model import generate_model
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import numpy as np
import seaborn as sns

# Data==========================================================================
data_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization values
])
train_path = r'C:\Users\Asus\Desktop\Work\低颅压\data\train'
test_path = r'C:\Users\Asus\Desktop\Work\低颅压\data\test'
weight_path = r'C:\Users\Asus\Desktop\Work\低颅压\MedicalNet\program_resnetV2\weight'
best_weight = os.path.join(weight_path, 'best_model_medicalnet_10.pth')
train_dirs, test_dirs = img_load_files(train_path, test_path)
train_dataset = MyDataSet(train_path, train_dirs, Cross_Entropy=True)
test_dataset = MyDataSet(test_path, test_dirs, Cross_Entropy=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# ==============================================================================

# Model=========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sets = parse_opts()
sets.input_D = 32
sets.input_H = 512
sets.input_W = 512
sets.pretrain_path = r'C:\Users\Asus\Desktop\Work\低颅压\MedicalNet\pretrain\resnet_10_23dataset.pth'
sets.model = 'resnet'
sets.gpu_id = [0]
sets.model_depth = 10
C = 2
torch.manual_seed(sets.manual_seed)
model, parameters = generate_model(sets)
model = model.to(device)
model.module.conv_seg = nn.Sequential(
    nn.Conv3d(in_channels=512, out_channels=C, kernel_size=1, stride=1, padding=0),
    nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global 3D average pooling
    nn.Flatten(),  # Flatten output to match the expected input of the fully connected layer
    nn.Linear(C, C),  # This is a fully connected layer
)

if os.path.exists(best_weight):
    model.load_state_dict(torch.load(best_weight))
    print('successful load weight！')
else:
    print('not successful load weight')

# Ensure model parameters are on the correct device
model = model.to(device)
# ==============================================================================

# Hyperparameter================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5, verbose=True)
num_epochs = 100
best_loss = float('inf')
losses_train = []
losses_test = []
accuracies_train = []
accuracies_test = []
# ==============================================================================

# main==========================================================================
model.eval()
loss_test_sum, accuracy_test = 0.0, 0.0
all_labels_test = []
all_outputs_test = []

with torch.no_grad():
    for images_test, labels_test, _ in test_loader:
        images_test = images_test.to(device)
        labels_test = labels_test.to(device)
        outputs_test = model(images_test)
        loss_test = criterion(outputs_test, labels_test)
        outputs_test_SM = F.softmax(outputs_test, dim=1)

        loss_test_sum += loss_test.item()
        predicted_labels_test = torch.round(outputs_test_SM)
        accuracy_test += torch.sum(predicted_labels_test == labels_test).item()

        all_labels_test.append(labels_test.cpu().numpy())
        all_outputs_test.append(outputs_test_SM.cpu().numpy())

avg_loss_test = loss_test_sum / len(test_loader)
avg_accuracy_test = 0.5 * accuracy_test / len(test_dataset)
losses_test.append(avg_loss_test)
accuracies_test.append(avg_accuracy_test)

all_labels_test = np.concatenate(all_labels_test, axis=0)
all_outputs_test = np.concatenate(all_outputs_test, axis=0)

# Function to plot confusion matrix, PR curve, and ROC curve
def plot_metrics(all_labels, all_outputs, class_labels):
    # Calculate metrics
    conf_matrix = confusion_matrix(np.argmax(all_labels, axis=1), np.argmax(all_outputs, axis=1))
    conf_matrix = conf_matrix[[1, 0], :][:, [1, 0]]  # Swap SIH and Non-SIH rows and columns
    precision = {}
    recall = {}
    pr_auc = {}
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(class_labels)):
        precision[i], recall[i], _ = precision_recall_curve(all_labels[:, i], all_outputs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[0], 
                xticklabels=['SIH', 'Non-SIH'], yticklabels=['SIH', 'Non-SIH'])
    axs[0].set_title('(a) Confusion Matrix')
    axs[0].set_xlabel('Predicted Labels')
    axs[0].set_ylabel('True Labels')

    # Plot PR curves
    for i in range(len(class_labels)):
        axs[1].step(recall[i], precision[i], where='post', label=f'{class_labels[i]} (AUC={pr_auc[i]:.2f})')
    axs[1].set_title('(b) PR Curves')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc='best')

    # Plot ROC curves
    for i in range(len(class_labels)):
        axs[2].plot(fpr[i], tpr[i], label=f'{class_labels[i]} (AUC={roc_auc[i]:.2f})')
    axs[2].plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line
    axs[2].set_title('(c) ROC Curves')
    axs[2].set_xlabel('False Positive Rate')
    axs[2].set_ylabel('True Positive Rate')
    axs[2].legend(loc='best')

    plt.tight_layout()
    plt.show()

class_labels = ['Non-SIH', 'SIH']
plot_metrics(all_labels_test, all_outputs_test, class_labels)

print(f'测试损失: {avg_loss_test:.4f}', f'测试准确率: {avg_accuracy_test:.4f}')
# ==============================================================================
