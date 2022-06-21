import os
import sys
import json
import copy
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image

import argparse
import timm
from tqdm import tqdm

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

sys.path.append('..')
from src.utils import *
from src.model import *

# setup seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 0
seed_everything(SEED)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load config
with open('config.json', 'r', encoding='utf8') as f:
    config_json = json.load(f)

data_path = config_json['data_path']
label_file = config_json['label_file']
landmark_file = config_json['landmark_file']

symptoms_list = ['Red_lid', 'Red_conj','Swl_crncl', 'Swl_lid', 'Swl_conj']

# load label info
labels_eye, labels_patient = load_labels(data_path, label_file)
landmarks_left, landmarks_right = load_landmark(data_path, landmark_file)



def model_run(setting, base, mode, symptom):
    # setup label info
    symptoms_left = ['_'.join([symptom, 'left', 'true']) for symptom in symptoms_list]
    symptoms_right = ['_'.join([symptom, 'right', 'true']) for symptom in symptoms_list]

    # make directory
    if not os.path.exists(os.path.join(setting)):
        os.mkdir(os.path.join(setting))

    if not os.path.exists(os.path.join(setting, base)):
        os.mkdir(os.path.join(setting, base))

    if setting == 'binary':
        if not os.path.exists(os.path.join(setting, base, symptom)):
            os.mkdir(os.path.join(setting, base, symptom))

        labels_left = labels_eye['_'.join([symptom, 'left', 'true'])].to_dict()
        labels_right = labels_eye['_'.join([symptom, 'right', 'true'])].to_dict()

    elif setting == 'multi':
        symptom = None

        labels_left = labels_eye.apply(lambda s: s[symptoms_left].astype(int).tolist(), axis=1).to_dict()
        labels_right = labels_eye.apply(lambda s: s[symptoms_right].astype(int).tolist(), axis=1).to_dict()

    # read splits
    with open('cv_list', 'rb') as f:
        cv_list = pickle.load(f)

    # run model training
    for k in range(10):
        cv_k = cv_list[k]

        pid_train = cv_k[1]
        pid_valid = cv_k[2]
        pid_test = cv_k[3]

        # k folds
        pid_train = pid_train.tolist() + pid_valid.tolist()
        pid_test = pid_test.tolist()

        np.random.shuffle(pid_train)
        np.random.shuffle(pid_test)

        results_all = []
        for kf in range(5):
            train_num = len(pid_train)
            valid_num = round(0.2 * len(pid_train))

            pid_valid_k = pid_train[kf*valid_num:(kf+1)*valid_num]
            pid_train_k = [pid for pid in pid_train if pid not in pid_valid_k]
            pid_test_k = pid_test

            img_train = [str(pid)+'_left' for pid in pid_train_k] + [str(pid)+'_right' for pid in pid_train_k]
            img_valid = [str(pid)+'_left' for pid in pid_valid_k] + [str(pid)+'_right' for pid in pid_valid_k]
            img_test = [str(pid)+'_left' for pid in pid_test_k] + [str(pid)+'_right' for pid in pid_test_k]

            # Load image
            IMGAE_SIZE = 224

            train_transform = transforms.Compose([
                              transforms.Resize((IMGAE_SIZE, IMGAE_SIZE)),
                              transforms.RandomCrop((IMGAE_SIZE, IMGAE_SIZE), padding=5),
                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            valid_transform = transforms.Compose([
                              transforms.Resize((IMGAE_SIZE, IMGAE_SIZE)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            train_dataset = img_Dataset(data_path, symptom, img_train, landmarks_left, landmarks_right,
                                        labels_left, labels_right, mode, train_transform)
            valid_dataset = img_Dataset(data_path, symptom, img_valid, landmarks_left, landmarks_right,
                                        labels_left, labels_right, mode, valid_transform)
            test_dataset = img_Dataset(data_path, symptom, img_test, landmarks_left, landmarks_right,
                                       labels_left, labels_right, mode, valid_transform)

            BATCH_SIZE = 32

            train_batch = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=16)
            valid_batch = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=16)
            test_batch = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=16)

            # Define model
            model = Prediction_model(setting, base)
            # model.apply(init_weights)
            model.to(device)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

            best_loss = np.inf
            prev_val_loss = np.inf
            counter = 0

            for e in range(50):
                if setting == 'binary':
                    train_loss, train_metrics = train_binary(model, optimizer, criterion, train_batch, device)
                    valid_loss, valid_metrics = evaluate_binary(model, criterion, valid_batch, device)
                    test_loss, test_metrics = evaluate_binary(model, criterion, test_batch, device)
                elif setting == 'multi':
                    train_loss, train_metrics = train_multi(model, optimizer, criterion, train_batch, device)
                    valid_loss, valid_metrics = evaluate_multi(model, criterion, valid_batch, device)
                    test_loss, test_metrics = evaluate_multi(model, criterion, test_batch, device)

                # if e % 5 == 0:
                print("[Epoch: %03d] train loss : %3.3f | %3.3f | %3.3f | %3.3f | %3.3f" % (e+1, train_loss, train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3]))
                print("[Epoch: %03d] valid loss : %3.3f | %3.3f | %3.3f | %3.3f | %3.3f" % (e+1, valid_loss, valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]))
                print("[Epoch: %03d] test  loss : %3.3f | %3.3f | %3.3f | %3.3f | %3.3f" % (e+1, test_loss, test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]))

                scheduler.step()

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                if valid_loss < prev_val_loss:
                    counter = 0
                else:
                    counter += 1

                if counter >= 5:
                    break

                prev_val_loss = valid_loss
                # model.load_state_dict(best_model_wts)

            best_model = copy.deepcopy(model)
            best_model.load_state_dict(best_model_wts)

            ## get all performance metrics
            best_model.eval()
            total_loss = 0
            trues_list, preds_list, scores_list = [], [], []
            with torch.no_grad():
                for batch in test_batch:
                    x = batch['img'].to(device)
                    y = batch['label'].float().to(device)

                    logit = model(x)

                    scores = torch.sigmoid(logit)
                    preds = scores.round()
                    trues = y.round()

                    trues_list.append(trues.cpu().detach().numpy())
                    preds_list.append(preds.cpu().detach().numpy())
                    scores_list.append(scores.cpu().detach().numpy())

            trues_all = np.vstack(trues_list)
            preds_all = np.vstack(preds_list)
            scores_all = np.vstack(scores_list)

            out = [k, kf, trues_all, preds_all, scores_all]
            results_all.append(out)

        # save out
        if base == 'TAOD-Net':
            out_path = os.path.join(base, symptom, '_'.join([str(mode), str(k)]))
            with open(out_path, 'wb') as f:
                pickle.dump(results_all, f)
            
        elif setting == 'binary':
            out_path = os.path.join(setting, base, symptom, '_'.join([str(mode), str(k)]))
            with open(out_path, 'wb') as f:
                pickle.dump(results_all, f)

        elif setting == 'multi':
            out_path = os.path.join(setting, base, '_'.join([str(mode), str(k)]))
            with open(out_path, 'wb') as f:
                pickle.dump(results_all, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TAOD-Net')

    parser.add_argument('--setting', type=str, default='multi')
    parser.add_argument('--base', type=str, default='ViT')
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--symptom', type=int, default=0)

    args = parser.parse_args()
    setting = args.setting
    base = args.base
    mode = args.mode
    symptom = symptoms_list[args.symptom]

    if setting not in ['binary', 'multi']:
        raise ValueError()

    if base not in ['ResNet50', 'SE-Net50', 'SK-Net50', 'ResNeSt50', 'EfficientNet', 'ViT', 'Swin', 'TAOD-Net']:
        raise ValueError()

    if mode not in range(8):
        raise ValueError()

    if symptom not in symptoms_list:
        raise ValueError()

    print('====================')
    print('-setting: {}'.format(setting))
    print('-base: {}'.format(base))
    print('-mode: {}'.format(mode))
    if setting == 'binary':
        print('-symptom: {}'.format(symptom))
    elif setting == 'multi':
        print('-symptom: {}'.format('all'))
    print('====================')

    model_run(setting, base, mode, symptom)
