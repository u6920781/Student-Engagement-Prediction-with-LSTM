import torch
import os
from DataLoader import DataLoader
from torch.utils import data
from lstm_net import mil_regression
import matplotlib.pyplot as plt


def save_image(data, file_name, root_path='results/', leg='leg', c = 'red'):
    save_path = str(root_path + file_name +".png")
    plt.plot(data, color=c, alpha=0.8, label=leg)
    # plt.legend()
    # plt.title("the Engagement Scores for " + file_name)
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    print(file_name + ' result saved ')
    plt.close()


class Visualize():
    def __init__(self, net, par_path, file_dir, seg_num):
        self.net = net
        self.par_path = par_path
        self.file_dir = file_dir
        self.seg_num = seg_num
        self.feature_num = net.feature_num

        self.net.load_state_dict(torch.load(self.par_path, map_location=torch.device('cpu')))
        dataset = DataLoader(root=file_dir, seg_num=self.seg_num)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

        for idx, (inputs, file_name) in enumerate(loader):
            # print(file_name)

            data = torch.zeros((1, self.seg_num, self.feature_num))  # batch_size, seg_num, features_num
            for i in range(self.seg_num):
                for j in range(1):
                    data[j, i, :] = inputs[i][j]
            # print(inputs)
            # print(data)
            # break
            output = net(data)
            scores = output.tolist()
            title = str(file_name)[2:-3]
            save_image(scores, title)


if __name__ == '__main__':
    net = mil_regression(feature_num=18, seg_num=60)
    path = 'net_parameters_case.pkl'
    file_dir = 'data_samples/OpenFace_features'
    Visualize(net, path, file_dir, 60)
