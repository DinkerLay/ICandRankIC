import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from algs.units import corr
from algs.models import group_net
from algs.losses import ListMLE_loss
from glob import glob
from tqdm import tqdm
import copy
import os

torch.random.manual_seed(2021)

# %%
month_list = glob('/home/dinker/codes/datas/factor_data/*.csv')
month_list.sort()


# %%
def read_month_data(path):
    df = pd.read_csv(path, index_col=0)
    df_ = df.iloc[:, 3:-3]
    return (torch.tensor(df_.values[:, :-1], dtype=torch.float32),
            torch.tensor(df_.values[:, -1], dtype=torch.float32),
            df.loc[:, ['ticker', 'tradeDate', 'real_return', 'market_value']])


def get_train_val_test_data(month_path, train_window=3):
    begin_ind = month_list.index(month_path)
    train_paths = month_list[begin_ind: begin_ind + train_window]
    val_path = month_list[begin_ind + train_window]
    test_path = month_list[begin_ind + train_window + 1]
    return [read_month_data(p) for p in train_paths], \
           read_month_data(val_path), \
           read_month_data(test_path)


# %%
r_ind = 17
train_data, val_data, test_data = get_train_val_test_data(month_list[r_ind], train_window=6)


# %%
def get_random_sample(train_data, batch_size=256):
    N_days = len(train_data)
    # pick day random
    rx, ry, _ = train_data[np.random.randint(low=0, high=N_days)]
    N_sample = ry.shape[0]
    # without repalcement sample
    # pick data random
    r_inds = np.random.choice(N_sample, batch_size, replace=False)
    return rx[r_inds], ry[r_inds]


def my_dataloader(train_data, batch_size=256, n_round=10):
    for _ in range(n_round):
        yield get_random_sample(train_data, batch_size=batch_size)


# %%
def train_model(train_td):
    model.train()
    data_loader = my_dataloader(train_td, batch_size=256, n_round=50)
    losses, flag = 0, 0
    for x_i, y_i in tqdm(data_loader):
        sig = model(x_i)
        loss = loss_func(sig, y_i.view_as(sig))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses += loss.item()
        flag += 1
    return losses / flag


def val_model(val_data):
    model.eval()
    sig = model(val_data[0])
    score = corr(sig, val_data[1])
    return score[0, 1]


def eval_model(test_data, model):  # top stk - bot stk
    model.eval()
    pred_ = model(test_data[0]).view(-1)
    sort_p = test_data[1][pred_.argsort()]
    return (sort_p[-100:] - sort_p[:100]).mean(), pred_.data.numpy()


# %%

train_window = 12
patience = 20
if not os.path.exists('./results/group_norm'): os.mkdir('./results/group_norm')
for ind in range(len(month_list) - train_window - 1):
    date = month_list[ind + train_window + 1][-11:]
    train_data, val_data, test_data = get_train_val_test_data(month_list[ind], train_window=train_window)
    model = group_net(244, drop_out=0.3)
    loss_func = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # train_td = TensorDataset(torch.cat([i[0] for i in train_data], axis=0),
    #                          torch.cat([i[1] for i in train_data], axis=0))

    max_ndcg = -np.inf
    p_flag = 0
    for i in range(200):
        train_loss = train_model(train_data)
        eval_acc = val_model(val_data)
        ret_test, _ = eval_model(test_data, model)
        print('train loss: %.4f | val ic@100: %.4f | test return: %.4f' % (train_loss,
                                                                           eval_acc, ret_test))
        if eval_acc > max_ndcg:
            max_ndcg = eval_acc
            best_model = copy.deepcopy(model)
            p_flag = 0
        p_flag += 1
        if p_flag > patience:
            break
    _, pred_ = eval_model(test_data, best_model)
    if os.path.exists('./results/group_norm/%s' % date):
        save_data = pd.read_csv('./results/group_norm/%s' % date, index_col=0)
        num_flag = int(save_data.columns[-1].split('_')[-1])
        save_data['pred_%s' % (num_flag + 1)] = pred_
    else:
        test_data[2]['pred_1'] = pred_
        save_data = test_data[2]
    save_data.to_csv('./results/group_norm/%s' % date)
    print('========== %s has done' % date)


