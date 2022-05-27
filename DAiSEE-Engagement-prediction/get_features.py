import numpy as np
import torch


class Features():
    def __init__(self, args, file_dir='data_samples/Test/5000441001.csv'):
        self.file_dir = file_dir
        self.features = []
        self.clipNums = args.clip_num
        self.args = args
        data = np.genfromtxt(file_dir, delimiter=',', dtype='str')
        data = np.delete(data, 0, 0)  # delete table caption
        v_data = data.astype(np.float)
        v_au_data = v_data[:, -35:-19]
        v_eye_data = v_data[:, 5:11]
        v_pose_data = v_data[:, -81:-75]

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
            v_data = v_data[~np.isnan(v_data).any(axis=1)]

        self.interval = int(v_data.shape[0] / self.args.clip_num)

        self.process(v_data)

    def process(self, v_data):
        interval = self.interval
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
                head = np.std(seg[:, -6:], axis=0)
                combine = np.concatenate((gaze, head), axis=0)

            self.features.append(torch.FloatTensor(combine))

    def get(self):
        return self.features


if __name__ == '__main__':
    Features()
