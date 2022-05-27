import torch
import os
import csv
import pickle
import numpy as np
from torch.utils import data
import argparse


# /raid/BN/engagementwild2020/abhinavdhall_EngagementWild_2020/OpenFace_features/validation
# data_samples/OpenFace_features/

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--case_id', '-id', type=str, default='case-0')
    parser.add_argument('--clip_num', '-clip', type=int, default=10, help='segmentation number of one video')
    parser.add_argument('--eye_gaze', '-eye', type=int, default=1)
    # parser.add_argument('--pose_T', '-pT', type=int, default=1)
    parser.add_argument('--head_pose', '-head', type=int, default=1)
    parser.add_argument('--action_unit', '-au', type=int, default=1)

    args = parser.parse_args()

    return args


class Processing:
    ''' Load dataset as torch.tensor '''

    def __init__(self, args, case='test',
                 root='data_samples/OpenFace_features/',
                 l_dir='data_samples/Engagement_Labels_Engagement.csv'):
        self.case = case
        self.file_list = []
        self.label_list = []
        self.level = [0., 0.33, 0.66, 1.]
        self.args = args

        # Only run for the first time
        for sample in os.listdir(root):
            if sample != "subject_45_Vid_7.txt":
                target_file = os.path.join(root, sample)
                self.file_list.append(target_file)
            with open(l_dir, 'r') as myfile:
                lines = csv.reader(myfile)
                for line in lines:
                    if line[0] == sample.strip('.txt'):
                        self.label_list.append(float(line[1]))
                        break
        # assert len(self.file_list) == len(self.label_list)
        self.all_features = self.get_feature()

        # Load data to speed up
        # with open(case + '_features_case5.txt', 'rb') as fp:
        #     self.all_features = pickle.load(fp)
        # with open(case + '_label_case5.txt', 'rb') as fp:
        #     self.label_list = pickle.load(fp)
        # print(case + ' dataset loaded successfully!')

    def get_feature(self):
        features = []
        for idx in range(len(self.label_list)):
            # segment video to 10 segments, return combine
            file_dir, label = self.file_list[idx], self.label_list[idx]
            print(file_dir)
            v_data = np.genfromtxt(file_dir, delimiter=',', dtype='str')
            v_data = np.delete(v_data, 0, 0)  # delete table caption
            v_data = v_data.astype(np.float)
            # v_data = v_data[~np.isnan(v_data).any(axis=1)]  # remove nan

            # v_data = v_data.astype(np.float)[:, (-35, -34, -33, -32, -30, -26, -24, -22, -20)]  # 01,02,04,05,07,14,17,23,26
            v_au_data = v_data[:, -35:-19]
            # # v_au_data = v_data.astype(np.float)[:, (-35, -34, -33, -32, -30, -26, -24, -22, -20)]
            v_eye_data = v_data[:, 4:10]
            v_pose_data = v_data[:, 10:16]   # pose
            # v_poseT_data = v_data[:, 10:13]  # poseT
            # v_poseR_data = v_data[:, 13:16]  # poseR


            # interval = int(v_data.shape[0] / self.args.clip_num)
            # feature = []

            if self.args.action_unit == 1:
                if self.args.eye_gaze == 1 and self.args.head_pose == 0:
                    v_data = np.concatenate((v_eye_data, v_au_data), axis=1)
                    v_data = v_data[~np.isnan(v_data).any(axis=1)]
                elif self.args.eye_gaze == 0 and self.args.head_pose == 1:
                    v_data = np.concatenate((v_pose_data, v_au_data), axis=1)
                    v_data = v_data[~np.isnan(v_data).any(axis=1)]
                elif self.args.eye_gaze == 1 and self.args.head_pose == 1:
                    v_data = np.concatenate((v_eye_data, v_pose_data), axis=1)
                    v_data = np.concatenate((v_data, v_au_data), axis=1)
                    v_data = v_data[~np.isnan(v_data).any(axis=1)]
                elif self.args.eye_gaze == 0 and self.args.head_pose == 0:
                    v_data = v_au_data
                    v_data = v_data[~np.isnan(v_data).any(axis=1)]
            else:
                v_data = np.concatenate((v_eye_data, v_pose_data), axis=1)
                # v_data = np.concatenate((v_data, v_poseR_data), axis=1)
                v_data = v_data[~np.isnan(v_data).any(axis=1)]

            interval = int(v_data.shape[0] / self.args.clip_num)
            feature = []

            for i in range(self.args.clip_num):
                combine = []
                if self.args.action_unit == 1:
                    if self.args.eye_gaze == 1 and self.args.head_pose == 0:
                        m = np.mean(v_data, axis=0)
                        seg = v_data[i * interval:(i + 1) * interval, :]
                        gaze = np.mean(np.abs(seg[:, :6] - m[:6]), axis=0)

                        max_intensity = np.max(seg, axis=0)[6:]
                        standard_deviation = np.std(seg, axis=0)[6:]
                        variation = seg - m
                        max_variation = np.max(variation, axis=0)[6:]
                        combine = np.concatenate((combine, max_intensity), axis=0)  # dim + 16
                        combine = np.concatenate((combine, max_variation), axis=0)  # dim + 16
                        combine = np.concatenate((combine, standard_deviation), axis=0)  # dim + 16
                        combine = np.concatenate((combine, gaze), axis=0)  # dim +6

                    elif self.args.eye_gaze == 0 and self.args.head_pose == 1:
                        m = np.mean(v_data, axis=0)
                        seg = v_data[i * interval:(i + 1) * interval, :]
                        head = np.std(seg[:, :6], axis=0)

                        max_intensity = np.max(seg[:, 6:], axis=0)
                        standard_deviation = np.std(seg[:, 6:], axis=0)
                        variation = seg - m
                        max_variation = np.max(variation, axis=0)[6:]
                        combine = np.concatenate((combine, max_intensity), axis=0)  # dim + 16
                        combine = np.concatenate((combine, max_variation), axis=0)  # dim + 16
                        combine = np.concatenate((combine, standard_deviation), axis=0)  # dim + 16
                        combine = np.concatenate((combine, head), axis=0)  # dim +6


                    elif self.args.eye_gaze == 1 and self.args.head_pose == 1:
                        m = np.mean(v_data, axis=0)
                        seg = v_data[i * interval:(i + 1) * interval, :]
                        gaze = np.mean(np.abs(seg[:, :6] - m[:6]), axis=0)
                        head = np.std(seg[:, 6:12], axis=0)
                        combine = np.concatenate((gaze, head), axis=0)

                        max_intensity = np.max(seg, axis=0)[12:]
                        standard_deviation = np.std(seg, axis=0)[12:]
                        variation = seg - m
                        max_variation = np.max(variation, axis=0)[12:]
                        combine = np.concatenate((combine, max_intensity), axis=0)  # dim + 16
                        combine = np.concatenate((combine, max_variation), axis=0)  # dim + 16
                        combine = np.concatenate((combine, standard_deviation), axis=0)  # dim + 16

                    elif self.args.eye_gaze == 0 and self.args.head_pose == 0:
                        m = np.mean(v_data, axis=0)
                        seg = v_data[i * interval:(i + 1) * interval, :]

                        max_intensity = np.max(seg, axis=0)
                        standard_deviation = np.std(seg, axis=0)
                        variation = seg - m
                        max_variation = np.max(variation, axis=0)
                        combine = np.concatenate((combine, max_intensity), axis=0)  # dim + 16
                        combine = np.concatenate((combine, max_variation), axis=0)  # dim + 16
                        combine = np.concatenate((combine, standard_deviation), axis=0)  # dim + 16


                elif self.args.action_unit == 0:
                    m = np.mean(v_data, axis=0)
                    seg = v_data[i * interval:(i + 1) * interval, :]

                    gaze = np.mean(np.abs(seg[:, :6] - m[:6]), axis=0)
                    # headT = np.mean(np.abs(seg[:, -6:-3] - m[-6:-3]), axis=0)
                    head = np.std(seg[:, -6:], axis=0)
                    combine = np.concatenate((gaze, head), axis=0)
                    # combine = np.concatenate((combine, headR), axis=0)

                feature.append(torch.FloatTensor(combine))
            features.append(feature)



        with open('data/' + self.case + '_features_' + self.args.case_id + '.txt', 'wb') as fp:
            pickle.dump(features, fp)
        with open('data/' + self.case + '_label_' + self.args.case_id + '.txt', 'wb') as fp:
            pickle.dump(self.label_list, fp)
        return features



if __name__ == '__main__':
    # test
    args = parse_args()
    test_root = '/raid/BN/engagementwild2020/abhinavdhall_EngagementWild_2020/OpenFace_features/validation'
    train_root = '/raid/BN/engagementwild2020/abhinavdhall_EngagementWild_2020/OpenFace_features/Train'
    l_dir = 'Engagement_Labels_Engagement.csv'
    Processing(root=test_root, case='test', l_dir=l_dir, args=args)
    Processing(root=train_root, case='train', l_dir=l_dir, args=args)
