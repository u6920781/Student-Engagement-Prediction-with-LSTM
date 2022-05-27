import torch
import os
import csv
import pickle
import numpy as np
from torch.utils import data
from get_features import Features
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--case_id', '-id', type=str, default='case-10-eye-pose')
    parser.add_argument('--clip_num', '-clip', type=int, default=5, help='segmentation number of one video')
    parser.add_argument('--eye_gaze', '-eye', type=int, default=1)
    parser.add_argument('--head_pose', '-head', type=int, default=1)
    parser.add_argument('--action_unit', '-au', type=int, default=0)

    args = parser.parse_args()

    return args


class OpenfaceDataset(torch.utils.data.Dataset):

    def __init__(self, args, case='Test', initialize=False, data_root='sample_data', label_root='sample_data/Labels'):
        self.case = case
        self.file_list = []
        self.all_labels = []
        self.all_features = []
        self.seg_num = args.clip_num
        self.args = args

        if initialize:
            self.initialize(data_root, label_root)
        else:
            self.load_data()

    def initialize(self, data_root='sample_data', label_root='sample_data/Labels'):
        case = self.case
        label_case = case + 'Labels.csv'

        for sample in os.listdir(os.path.join(data_root, case)):
            target_file = os.path.join(data_root, case, sample)
            with open(os.path.join(label_root, label_case), 'r') as myfile:
                lines = csv.reader(myfile)
                for line in lines:
                    if line[0][:-4] == sample[:-4]:
                        self.file_list.append(target_file)
                        self.all_labels.append(float(line[2]))  # engagement
                        break

        assert len(self.file_list) == len(self.all_labels)
        self.all_features = self.get_feature()
        self.save_features()

    def save_features(self):
        with open('openface_data/' + self.case + '_features_' + self.args.case_id + '.txt', 'wb') as fp:
            pickle.dump(self.all_features, fp)
        with open('openface_data/' + self.case + '_label_' + self.args.case_id + '.txt', 'wb') as fp:
            pickle.dump(self.all_labels, fp)
        print(self.case + ' dataset saved successfully!')

    def get_feature(self):
        features = []
        for idx in range(len(self.all_labels)):
            file_dir = self.file_list[idx]
            feature = Features(self.args, file_dir).get()
            features.append(feature)
            print(file_dir)
        return features

    def load_data(self):
        with open('openface_data/' + self.case + '_features_' + self.args.case_id + '.txt', 'rb') as fp:
            self.all_features = pickle.load(fp)
        with open('openface_data/' + self.case + '_label_' + self.args.case_id + '.txt', 'rb') as fp:
            self.all_labels = pickle.load(fp)
        print(self.case + ' dataset loaded successfully!')

    def __getitem__(self, idx):
        x, y = self.all_features[idx], self.all_labels[idx]
        return x, y

    def __len__(self):
        return len(self.all_labels)


if __name__ == '__main__':
    args = parse_args()
    # OpenfaceDataset(args=args, initialize=True)
    #
    OpenfaceDataset(args, case='Train', initialize=True, data_root='/raid/xinrantian/DAiSEE/OpenFace_features',
                    label_root='/raid/xinrantian/DAiSEE/OpenFace_features/Labels')

    OpenfaceDataset(args, case='Test', initialize=True, data_root='/raid/xinrantian/DAiSEE/OpenFace_features',
                    label_root='/raid/xinrantian/DAiSEE/OpenFace_features/Labels')

    # OpenfaceDataset(case='Validation', initialize=True, data_root='/raid/xinrantian/DAiSEE/OpenFace_features',
    #                 label_root='/raid/xinrantian/DAiSEE/OpenFace_features/Labels', case_num=2)
