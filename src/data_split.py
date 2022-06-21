import os
import json
import pickle
import random

import numpy as np
import pandas as pd
import torch

from skmultilearn.model_selection import iterative_train_test_split

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

# load config
with open('config.json', 'r', encoding='utf8') as f:
    config_json = json.load(f)

data_path = config_json['data_path']
label_file = config_json['label_file']
landmark_file = config_json['landmark_file']

labels = pd.read_excel(os.path.join(data_path, label_file))
labels = labels.set_index('p_num')
labels = labels.drop(349) # image error


# check symptom information
symptoms_list = ['Red_lid', 'Red_conj','Swl_crncl', 'Swl_lid', 'Swl_conj']

labels_revised = []
for symptom in symptoms_list:
    for loc in ['left', 'right']:
        c_names = [c for c in labels.columns if (symptom in c) & (loc in c) & ('MG' not in c)]
        soft_label = labels[c_names].mean(axis=1)
        hard_label = labels[c_names].mean(axis=1).round(0)

        c_names_MG = [c for c in labels.columns if (symptom in c) & (loc in c) & ('MG' in c)]
        true_label = labels[c_names_MG].squeeze()

        soft_label.name = '_'.join([symptom, loc, 'soft'])
        hard_label.name = '_'.join([symptom, loc, 'hard'])
        true_label.name = '_'.join([symptom, loc, 'true'])

        labels_revised.append(soft_label)
        labels_revised.append(hard_label)
        labels_revised.append(true_label)

        consensus = (soft_label == true_label)
        consensus.name = '_'.join([symptom, loc, 'consensus'])

        labels_revised.append(consensus)

labels_revised = pd.concat(labels_revised, axis=1)
labels_revised = labels_revised.dropna()

# check patient-level symptom
labels_patient = []
for symptom in ['Red_lid', 'Red_conj', 'Swl_crncl', 'Swl_lid', 'Swl_conj']:
    patient_label = labels_revised[[c for c in labels_revised.columns if (symptom in c) & ('true' in c)]].sum(axis=1)
    patient_label = (patient_label >= 1).astype(int)

    patient_label.name = '_'.join([symptom, 'patient'])
    labels_patient.append(patient_label)

labels_patient = pd.concat(labels_patient, axis=1)
labels_patient = labels_patient.dropna()

# filter CAS score nan
nan_idx = np.unique(labels.loc[labels['symptom_1'].isna()].index.to_list() + labels.loc[labels['symptom_2'].isna()].index.to_list())
nan_idx = [idx for idx in nan_idx if idx in labels_patient.index]
cas_idx = [idx for idx in labels_patient.index if idx not in nan_idx]


cv_list = []
for i in range(20):
    labels_split = labels_patient.copy()
    labels_split = labels_split.loc[cas_idx]
    labels_split = labels_split.iloc[:, :5]

    X_train, y_train, X_test, y_test = iterative_train_test_split(labels_split.reset_index()[['p_num']].to_numpy(), labels_split.to_numpy(), test_size = 100/len(labels_split))

    X_train_all = np.hstack([X_train.flatten(), nan_idx])
    labels_train = labels_patient.copy().loc[X_train_all]
    labels_train = labels_train.iloc[:, :5]

    X_train, y_train, X_valid, y_valid = iterative_train_test_split(labels_train.reset_index()[['p_num']].to_numpy(), labels_train.to_numpy(), test_size = 100/len(labels_train))

    cv_list.append([i, X_train.flatten(), X_valid.flatten(), X_test.flatten()])

with open('cv_list', 'wb') as f:
    pickle.dump(cv_list, f)

print('==================== Save {} ===================='.format('cv_list'))

