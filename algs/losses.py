import torch
import torch.nn as nn
import numpy as np


# %%
class quantile_loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_true = y_true.view_as(y_pred)
        y_sort, indices_a = y_true.sort(descending=False, dim=0)
        n = indices_a.shape[0]
        y_sort_pred = y_pred[indices_a]
        w = torch.arange(1, n + 1).float().view_as(y_pred) / n
        loss = self.loss_func(y_sort_pred.view_as(y_pred), w)
        return loss.mean()


class ListMLE_loss(nn.Module):

    def __init__(self, eps=1e-5):
        super(ListMLE_loss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_sort, indices = y_true.sort(descending=True, dim=0)
        indices = indices.view(-1)
        y_sort_pred = y_pred[indices]
        cumloss = torch.cumsum(y_sort_pred.exp().flip(dims=[0]), dim=0).flip(dims=[0])
        observed_loss = torch.log(cumloss + self.eps) - y_sort_pred
        return observed_loss.sum()


class RankNet_loss(nn.Module):
    '''
    the y_true need to be sorted
    '''

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.view_as(y_pred)
        _, indices_a = y_true.sort(descending=True, dim=0)
        y_sort = y_pred[indices_a.view(-1)]
        y_m = y_sort - y_sort.reshape(1, -1)
        log_reg = torch.log(1 + torch.exp(y_m))
        tri_m = torch.triu(torch.ones_like(y_m, device=y_pred.device), diagonal=0)
        base_loss = y_m * tri_m
        return -base_loss.sum() + log_reg.sum()


class ListLs_loss(nn.Module):

    def __init__(self, eps=1e-5):
        super(ListLs_loss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_sort, indices_a = y_true.sort(descending=True, dim=0)
        indices_a = indices_a.view(-1)
        y_sort_pred = y_pred[indices_a]
        y_exp = y_sort_pred.exp()
        cumloss_inv = torch.cumsum(y_exp.flip(dims=[0]), dim=0).flip(dims=[0])
        observed_long = torch.log(cumloss_inv + self.eps)

        y_sort, indices_b = y_true.sort(descending=False, dim=0)
        indices_b = indices_b.view(-1)
        y_sort_pred = y_pred[indices_b]
        y_iexp = (-y_sort_pred).exp()
        cumloss_inv = torch.cumsum(y_iexp.flip(dims=[0]), dim=0).flip(dims=[0])
        observed_short = torch.log(cumloss_inv + self.eps)

        return (observed_long + observed_short).mean()

class RankNet_approx_loss(nn.Module):
    '''
    the y_true need to be sorted
    '''

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.view_as(y_pred)
        y_sort, indices_a = y_true.sort(descending=True, dim=0)
        y_inverse, indices_b = y_true.sort(descending=False, dim=0)
        indices_a, indices_b = indices_a.view(-1), indices_b.view(-1)
        n = indices_a.shape[0]
        y_sort_pred = y_pred[indices_a]
        y_sort_inverse = y_pred[indices_b]
        w = (n - torch.arange(0, n).float()).view_as(y_pred) / n
        loss = - w * (y_sort_pred - y_sort_inverse)
        return loss.mean()

class Fang_loss(nn.Module):
    
    def __init__(self, lam = 1.83, eps = 1e-6):
        super(Fang_loss, self).__init__()
        self.trans_func = lambda x: 1 / ( 1 + torch.exp( -lam / 2 * (x - x.mean()) / x.std() ) )
        self.norm = lambda x: (x - x.mean()) / (x.std() + eps)
        
    def forward(self, y_pred, y_true):
        y_true =  y_true.view_as(y_pred)
        y_pred_tr = self.trans_func(y_pred)
        y_true_tr = self.trans_func(y_true)
        loss = - self.norm(y_pred_tr) * self.norm(y_true_tr)
        return loss.mean()

class Zhang_loss(nn.Module):
    
    def __init__(self, eps=1e-5):
        super(Zhang_loss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_sort, indices = y_true.sort(descending=True, dim=0)
        indices = indices.view(-1)
        cn = int(indices.shape[0] / 2)
        y_sort_pred = (y_pred[indices] - y_pred[indices.flip(dims = [0])])[:cn]
        cumloss = torch.cumsum(y_sort_pred.exp().flip(dims=[0]), dim=0).flip(dims=[0])
        observed_loss = torch.log(cumloss + self.eps) - y_sort_pred
        return observed_loss.mean()

def ic_reg(x, y, lam = 0.1):
    y = y.view_as(x)
    mse = torch.mean(torch.square(y - x))
    ic = corr_torch(x, y)
    return mse - lam * ic


def rank_reg(x, y, lam = 0.1):
    y = y.view_as(x)
    y_m = y - y.reshape(1, -1)
    x_m = x - x.reshape(1, -1)
    p = - y_m * x_m
    mse = ((y - x) ** 2).mean()
    p_reg = nn.functional.relu(p).mean()

    return mse + lam * p_reg

def corr_torch(x, y):
    x_mean = x.mean()
    x_std = x.std()
    y_mean = y.mean()
    y_std = y.std()
    ic_x = (x - x_mean) / x_std
    ic_y = (y - y_mean) / y_std
    return (ic_x * ic_y).mean()

def rescale_ic(x, y):
    y = y.view_as(x)
    x_ = torch.softmax(x, dim = 0)
    ic = corr_torch(x_, y)
    return -ic

