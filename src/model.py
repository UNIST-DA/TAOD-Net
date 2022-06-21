import numpy as np

import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
from tqdm import tqdm

from src.utils import *


# Define dataset
class img_Dataset(Dataset):
    def __init__(self, data_path, symptom, img_set, landmarks_left, landmarks_right, 
                 labels_left, labels_right, mode, image_transform):
        self.data_path = data_path
        self.symptom = symptom
        self.img_set = img_set
        self.landmarks_left = landmarks_left
        self.landmarks_right = landmarks_right
        self.labels_left = labels_left
        self.labels_right = labels_right
        self.mode = mode
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.img_set)
    
    def __getitem__(self, idx):
        pid = int(self.img_set[idx].split('_')[0])
        loc = self.img_set[idx].split('_')[1]
        
        if loc=='left':
            img_c = get_img_cropped(self.data_path, pid, self.landmarks_left, mode=self.mode, bound=0.2)
            img_c = self.image_transform(Image.fromarray(img_c))
            label = self.labels_left[pid]
        elif loc=='right':
            img_c = get_img_cropped(self.data_path, pid, self.landmarks_right, mode=self.mode, bound=0.2)
            img_c = cv2.flip(img_c, 1) # vertical flip
            img_c = self.image_transform(Image.fromarray(img_c))
            label = self.labels_right[pid]
        else:
            pass
        
        sample = {'label': torch.as_tensor(label), 'img': torch.FloatTensor(img_c)}
        return sample
    
    
# Define model
class Prediction_model(nn.Module):
    def __init__(self, setting='multi', base='ViT', hidden=512):
        super().__init__()

        # set model_dict
        model_dict = {
            'ResNet50': 'resnet50',
            'SE-Net50': 'legacy_seresnet50',
            'SK-Net50': 'skresnext50_32x4d',
            'ResNeSt50': 'resnest50d',
            'EfficientNet': 'efficientnet_b0',
            'ViT': 'vit_small_patch16_224',
            'Swin': 'swin_small_patch4_window7_224',
            'TAOD-Net': 'vit_small_patch16_224',
        }

        model_hidden_dict = {
            'ResNet50': 2048,
            'SE-Net50': 1000,
            'SK-Net50': 2048,
            'ResNeSt50': 2048,
            'EfficientNet': 1000,
            'ViT': 384,
            'Swin': 768,
            'TAOD-Net': 384,
        }

        # set target_num
        if setting=='binary':
            target_num = 1
        elif setting == 'multi':
            target_num = 5

        self.model = timm.create_model(model_dict[base], pretrained=True)
        if base in ['ResNet50', 'SE-Net50', 'SK-Net50', 'ResNeSt50', 'EfficientNet']:
            self.model.fc = nn.Identity()
        elif base in ['ViT', 'Swin', 'TAOD-Net']:
           self.model.head = nn.Identity()
        
        self.bn = nn.BatchNorm1d(model_hidden_dict[base])
        self.fc = nn.Sequential(nn.Linear(model_hidden_dict[base], hidden), 
                                nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(hidden, target_num))
                
        # Set all parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        x = self.fc(self.bn(x))
        return x

    
def train_binary(model, optimizer, criterion, train_iter, device):
    model.train()
    total_loss = 0
    trues_list, preds_list = [], []
    for batch in tqdm(train_iter):
        x = batch['img'].to(device)
        y = batch['label'].float().unsqueeze(1).to(device)
        optimizer.zero_grad()

        logit = model(x)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()
        
        preds = torch.sigmoid(logit).round().int()
        trues = y.round().int()

        total_loss += loss.item()
        trues_list.append(trues.cpu().detach().numpy())
        preds_list.append(preds.cpu().detach().numpy())
        
    trues_all = np.vstack(trues_list)
    preds_all = np.vstack(preds_list)
    
    avg_loss = total_loss / len(train_iter)
    avg_metrics = get_metrics(trues_all, preds_all)
    
    return avg_loss, avg_metrics 


def evaluate_binary(model, criterion, val_iter, device):
    model.eval()
    total_loss = 0
    trues_list, preds_list = [], []
    with torch.no_grad():
        for batch in val_iter:
            x = batch['img'].to(device)
            y = batch['label'].float().unsqueeze(1).to(device)

            logit = model(x)
            loss = criterion(logit, y)

            preds = torch.sigmoid(logit).round()
            trues = y.round()

            total_loss += loss.item()
            trues_list.append(trues.cpu().detach().numpy())
            preds_list.append(preds.cpu().detach().numpy())
            
    trues_all = np.vstack(trues_list)
    preds_all = np.vstack(preds_list)
        
    avg_loss = total_loss / len(val_iter)
    avg_metrics = get_metrics(trues_all, preds_all)
    
    return avg_loss, avg_metrics 
    
    
def train_multi(model, optimizer, criterion, train_iter, device):
    model.train()
    total_loss = 0
    trues_list, preds_list = [], []
    for batch in tqdm(train_iter):
        x = batch['img'].to(device)
        y = batch['label'].float().to(device)
        optimizer.zero_grad()

        logit = model(x)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()
        
        preds = torch.sigmoid(logit).round().int()
        trues = y.round().int()

        total_loss += loss.item()
        trues_list.append(trues.cpu().detach().numpy())
        preds_list.append(preds.cpu().detach().numpy())
        
    trues_all = np.vstack(trues_list)
    preds_all = np.vstack(preds_list)
    
    avg_loss = total_loss / len(train_iter)
    avg_metrics = []
    for i in range(5):
        avg_metrics.append(get_metrics(trues_all[:,i], preds_all[:,i]))
    avg_metrics = np.nanmean(avg_metrics, axis=0)
    
    return avg_loss, avg_metrics 


def evaluate_multi(model, criterion, val_iter, device):
    model.eval()
    total_loss = 0
    trues_list, preds_list = [], []
    with torch.no_grad():
        for batch in val_iter:
            x = batch['img'].to(device)
            y = batch['label'].float().to(device)

            logit = model(x)
            loss = criterion(logit, y)

            preds = torch.sigmoid(logit).round()
            trues = y.round()

            total_loss += loss.item()
            trues_list.append(trues.cpu().detach().numpy())
            preds_list.append(preds.cpu().detach().numpy())
            
    trues_all = np.vstack(trues_list)
    preds_all = np.vstack(preds_list)
        
    avg_loss = total_loss / len(val_iter)
    avg_metrics = []
    for i in range(5):
        avg_metrics.append(get_metrics(trues_all[:,i], preds_all[:,i]))
    avg_metrics = np.nanmean(avg_metrics, axis=0)
    
    return avg_loss, avg_metrics 
    
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)