import torch
import os
from torch.utils import data
from get_features import Features


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, root='data_samples/OpenFace_features', seg_num=10):
        self.root = root
        self.all_features = []
        self.seg_num = seg_num
        self.file_names = []

        for sample in os.listdir(root):
            if sample[-4:] == '.txt':
                self.file_names.append(sample[:-4])
                target_file = os.path.join(root, sample)
                features = Features(target_file, self.seg_num).get()
                self.all_features.append(features)

    def __getitem__(self, idx):
        x = self.all_features[idx]
        y = self.file_names[idx]
        return x, y

    def __len__(self):
        return len(self.file_names)


if __name__ == '__main__':
    DataLoader()
