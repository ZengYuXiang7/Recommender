# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import torch
import pickle

from torch.utils.data import DataLoader

from utils.config import get_config
from utils.logger import Logger
from utils.plotter import MetricsPlotter
from utils.utils import set_settings


import dgl
from scipy.sparse import csr_matrix


class experiment:
    def __init__(self, args):
        self.args = args

    def load_data(self, args):
        from scipy.sparse import csr_matrix
        import pandas as pd

        # 读取评分数据
        ratings = pd.read_csv("datasets/ml-25m/ratings.csv")

        # 将电影ID和用户ID转换为连续索引
        user_ids = ratings['userId'].unique()
        movie_ids = ratings['movieId'].unique()

        user_id_map = {id_: idx for idx, id_ in enumerate(user_ids)}
        movie_id_map = {id_: idx for idx, id_ in enumerate(movie_ids)}

        ratings['userId'] = ratings['userId'].map(user_id_map)
        ratings['movieId'] = ratings['movieId'].map(movie_id_map)

        # 创建用户-电影交互矩阵
        num_users = len(user_ids)
        num_movies = len(movie_ids)

        interaction_matrix = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])),
                                        shape=(num_users, num_movies))

        print("Interaction matrix shape:", interaction_matrix.shape)

        # 提取索引和评分
        dataX = ratings[['userId', 'movieId']].values
        dataY = ratings['rating'].values
        return dataX, dataY


    def preprocess_data(self, data, args):
        x, y = data
        return x, y


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.raw_data = exper_type.load_data(args)
        self.x, self.y = exper_type.preprocess_data(self.raw_data, args)
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, self.max_value = self.get_train_valid_test_dataset(self.x, self.y, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)} Max_value : {self.max_value:.2f}')

    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, args):
        return (
            TensorDataset(train_x, train_y, args),
            TensorDataset(valid_x, valid_y, args),
            TensorDataset(test_x, test_y, args)
        )

    def get_train_valid_test_dataset(self, x, y, args):
        x, y = np.array(x),  np.array(y)
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]

        # print(y.shape)
        max_value = y.max()
        y /= max_value

        train_size = int(len(x) * args.density)
        # train_size = int(args.train_size)
        valid_size = int(len(x) * 0.10)

        train_x = x[:train_size]
        train_y = y[:train_size]

        valid_x = x[train_size:train_size + valid_size]
        valid_y = y[train_size:train_size + valid_size]

        test_x = x[train_size + valid_size:]
        test_y = y[train_size + valid_size:]

        return train_x, train_y, valid_x, valid_y, test_x, test_y, max_value


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, args):
        self.args = args
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        useridx, itemidx = self.x[idx][0], self.x[idx][1]
        value = self.y[idx]
        return useridx, itemidx, value


def custom_collate_fn(batch, args):
    from torch.utils.data.dataloader import default_collate
    useridx, itemidx, value = zip(*batch)
    useridx = default_collate(useridx)
    itemidx = default_collate(itemidx)
    value = default_collate(value)
    return useridx, itemidx, value


def get_dataloaders(train_set, valid_set, test_set, args):
    # max_workers = multiprocessing.cpu_count()
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        # num_workers=max_workers,
        # prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.bs * 16,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        # num_workers=max_workers,
        # prefetch_factor=4
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.bs * 16,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        # num_workers=max_workers,
        # prefetch_factor=4
    )

    return train_loader, valid_loader, test_loader



if __name__ == '__main__':
    args = get_config()
    set_settings(args)
    args.experiment = False

    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Density : {args.density}"
    log_filename = f'{args.train_size}_r{args.dimension}'
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log
    log(str(args.__dict__))

    exper = experiment(args)
    datamodule = DataModule(exper, args)
    for train_batch in datamodule.train_loader:
        useridx, itemidx, value = train_batch
        print(useridx.shape, itemidx.shape, value.shape)
        break
    print('Done!')
