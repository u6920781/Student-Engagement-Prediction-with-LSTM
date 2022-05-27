import torch
import torch.nn as nn


class lstm_regression(nn.Module):
    def __init__(self, feature_num=18, hidden_dim=64):
        super(lstm_regression, self).__init__()
        self.feature_num = feature_num
        self.lstm = nn.LSTM(self.feature_num, hidden_dim, 1, batch_first=True)
        self.dense = torch.nn.Sequential(
            nn.Linear(hidden_dim, 1028),
            nn.Linear(1028, 512),
            nn.Linear(512, 128),
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
        # self.seg_num = seg_num
        self.net = lstm_regression(feature_num, hidden_dim)
        self.feature_num = feature_num
        self.seg_num = seg_num

    def forward(self, inputs):
        seg_num = self.seg_num
        outputs = torch.zeros(seg_num)
        for i in range(seg_num):
            outputs[i] = self.net(inputs[:, i, :]).squeeze()
        return outputs


if __name__ == '__main__':
    net = mil_regression()
    # net.load_state_dict(torch.load('net_parameters_case.pkl', map_location=torch.device('cpu')))
    # dataset = DataLoader()
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
    # for idx, (inputs,file_name) in enumerate(train_loader):
    #
    #     data = torch.zeros((1, 100, 18))  # batch_size, seg_num, features_num
    #     for i in range(100):
    #         for j in range(1):
    #             data[j, i, :] = inputs[i][j]
    #
    #     output = net(data)
    #     print(file_name)
    #     print(output)
