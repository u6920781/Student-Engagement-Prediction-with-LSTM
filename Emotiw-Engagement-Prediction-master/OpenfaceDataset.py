import torch
import os
import csv
import pickle
import numpy as np
from torch.utils import data

# OpenFace_features/Train/subject_26_Vid_1.txt
class OpenfaceDataset(torch.utils.data.Dataset):
    ''' Load dataset as torch.tensor '''
    def __init__(self, case='test', data_id='', root='OpenFace_features2/Train', l_dir='OpenFace_features/Labels.csv'):
        self.case = case
        self.file_list = []
        self.label_list = []
        self.level = [0., 0.33, 0.66, 1.]

        # Load data to speed up
        with open(self.case + '_features_' + data_id + '.txt', 'rb') as fp:
            self.all_features = pickle.load(fp)
        with open(self.case + '_label_' + data_id + '.txt', 'rb') as fp:
            self.label_list = pickle.load(fp)
        print(case+' dataset loaded successfully!')

    def __getitem__(self, idx):
        x, y = self.all_features[idx], self.label_list[idx]
        return x, y

    def __len__(self):
        return len(self.label_list)

#         return len(self.all_features)

if __name__ == '__main__':
    # test
    train_dataset = OpenfaceDataset(case='train')
    # test_dataset = OpenfaceTestset()

    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True)
    # for idx, (data, target) in enumerate(dataloader):
    #     tdata = torch.zeros((12, 10, 9))
    #     for i in range(10):
    #         for j in range(12):
    #             tdata[j][i] = data[i][j]
    #     print (tdata.shape)
    #     print target
    #     break