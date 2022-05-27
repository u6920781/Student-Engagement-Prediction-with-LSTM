import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from OpenFaceDataset import OpenfaceDataset


class lstm_regression(nn.Module):
    def __init__(self, feature_num=18, hidden_dim=64):
        super(lstm_regression, self).__init__()
        # self.lstm = nn.LSTM(feature_num, hidden_dim, 2, batch_first=True, dropout=0.5)
        self.lstm = nn.LSTM(feature_num, hidden_dim, 1, batch_first=True)
        self.dense = torch.nn.Sequential(
            nn.Linear(hidden_dim, 128),
            # nn.Linear(1028, 512),
            # nn.Linear(512, 128),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        output, _ = self.lstm(inputs)
        output = self.dense(output[:, -1, :])
        return output


class mil_regression(nn.Module):
    def __init__(self, feature_num=18, hidden_dim=64, seg_num=10):
        ''' use LSTM regression for MIL '''
        super(mil_regression, self).__init__()
        self.seg_num = seg_num
        self.net = lstm_regression(feature_num, hidden_dim)

    def forward(self, inputs):
        self.b, self.seg_num, _ = inputs.shape
        outputs = torch.zeros((self.b, self.seg_num)).cuda()
        for i in range(self.seg_num):
            outputs[:, i] = self.net(inputs[:, i, :]).squeeze()
        output = torch.mean(outputs, 1).cuda()
        return output


if __name__ == '__main__':
    net = mil_regression()
    # net = mil_regression()
    # dataset = OpenfaceDataset()
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
    # for idx, (inputs, target) in enumerate(train_loader):
    #     data = torch.zeros((1, 10, 18))  # batch_size, seg_num, features_num
    #     for i in range(10):
    #         for j in range(1):
    #             data[j, i, :] = inputs[i][j]
    #     output = net(data)
    #     print(output)
    #     break

