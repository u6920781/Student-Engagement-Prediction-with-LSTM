import numpy as np
import torch


class Features():
    def __init__(self, file_dir='data_samples/OpenFace_features/subject_20_Vid_2.txt', seg_num=100):
        self.file_dir = file_dir
        self.features = []
        self.clipNums = seg_num
        data = np.genfromtxt(file_dir, delimiter=',', dtype='str')
        data = np.delete(data, 0, 0)  # delete table caption

        data = data.astype(np.float)[:, (-35, -34, -33, -32, -30, -26, -24, -22, -20)]  # Case2

        self.data = data[~np.isnan(data).any(axis=1)]  # remove nan
        # print(self.data.shape)

        self.process()

    def process(self):
        m = np.mean(self.data, axis=0)
        interval = int(self.data.shape[0] / self.clipNums)

        for i in range(self.clipNums):
            seg = self.data[i * interval:(i + 1) * interval, :]
            max_intensity = np.max(seg, axis=0)
            standard_deviation = np.std(seg, axis=0)
            # variation = seg - m
            # max_variation = np.max(variation, axis=0)
            self.features.append(torch.FloatTensor(np.append(max_intensity, standard_deviation)))

    def get(self):
        return self.features


if __name__ == '__main__':
    Features()
