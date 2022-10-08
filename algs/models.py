import torch
import torch.nn as nn
import numpy as np


# %%
class NN2Batchnorm(nn.Module):

    def __init__(self, num_feature, hidden_size=64, drop_out=0.5,
                 affine=True, eta=1e-05):
        super(NN2Batchnorm, self).__init__()
        hidden_2 = int(np.sqrt(hidden_size))

        self.model = nn.Sequential(
            nn.Linear(num_feature, hidden_size),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_2),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_2, 1)
        )

        self.batch_norm = nn.BatchNorm1d(1, affine=affine)

    def forward(self, input_x):
        out = self.model(input_x)
        return self.batch_norm(out)


class NN2net(nn.Module):

    def __init__(self, num_feature, hidden_size=64, drop_out=0.5):
        super(NN2net, self).__init__()
        hidden_2 = int(np.sqrt(hidden_size))

        self.model = nn.Sequential(
            nn.Linear(num_feature, hidden_size),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_2),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_2, 1)
        )

    def forward(self, input_x):
        out = self.model(input_x)
        return out


def torch_corr(y_pred):
    mean = y_pred.mean(dim=1, keepdim=True).expand_as(y_pred)
    y_m = y_pred - mean
    n = y_m.shape[-1]
    cov = y_m @ y_m.t() / (n - 1)
    d = torch.diag(cov)
    stddev = torch.sqrt(d)
    cov /= stddev[:, None]
    cov /= stddev[None, :]
    return cov


class group_net(nn.Module):

    def __init__(self, num_feature, hidden_size=64, drop_out=0.5):
        super(group_net, self).__init__()
        hidden_2 = int(np.sqrt(hidden_size))

        self.model_vec = nn.Sequential(
            nn.Linear(num_feature, hidden_size),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_2),
        )

        self.model_pred = nn.Sequential(
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_2, 1)
        )

        self.batchnorm = nn.BatchNorm1d(1)
        self.batchnorm_h = nn.BatchNorm1d(hidden_2)

    def forward(self, input_x):
        out_vec = self.model_vec(input_x)
        out_vec = self.batchnorm_h(out_vec)
        p_mat = torch.softmax( torch.matmul(out_vec , out_vec.t()) , dim=-1)
        group_vec = torch.matmul(p_mat , out_vec)
        out = self.model_pred(group_vec)
        return self.batchnorm(out)  # , out_vec

    # def forward(self, input_x):
    #     out_vec = self.model_vec(input_x)
    #     w = torch_corr(out_vec)
    #     group_vec = w @ out_vec * out_vec
    #     group_vec = group_vec / group_vec.sum(dim=1, keepdims=True)
    #     out = self.model_pred(group_vec)
    #     return self.batchnorm(out)  # , group_vec

#     def forward(self, input_x):
#         out_vec = self.model_vec(input_x)
#         w = torch.softmax(torch_corr(out_vec), dim=0)
#         group_vec = w @ out_vec
#         out = self.model_pred(group_vec)
#         return self.batchnorm(out) #, out_vec


#     def forward(self, input_x):
#         out_vec = self.model_vec(input_x)
#         w = torch.softmax(torch_corr(out_vec), dim=1)
#         group_vec = w @ out_vec
#         out = self.model_pred(group_vec)
#         return self.batchnorm(out)  # , group_vec

# def forward(self, input_x):
#     out_vec = self.model_vec(input_x)
#     w = torch_corr(out_vec)
#     out = self.model_pred(out_vec)
#     out = w @ out
#     return out #, group_vec
