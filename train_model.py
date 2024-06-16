# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import pandas as pd
import torch

from tqdm import *

from data import experiment, DataModule

from modules.NeuCF import NeuCF
from utils.config import get_config
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.plotter import MetricsPlotter
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed
from utils.utils import makedir
global log, args
torch.set_default_dtype(torch.float32)


class Model(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.args = args
        self.input_size = data.x.shape[-1]
        self.hidden_size = args.dimension
        if args.model == 'neucf':
            self.model = NeuCF(args)

        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    def forward(self, useridx, itemidx):
        y = self.model.forward(useridx, itemidx)
        return y

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if args.classification else 'max', factor=0.5, patience=args.patience // 1.5, threshold=0.0)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader):
            useridx, itemidx, value = train_Batch
            useridx, itemidx, value = useridx.to(self.args.device), itemidx.to(self.args.device), value.to(self.args.device)
            pred = self.forward(useridx, itemidx)
            # print(pred.shape, value.shape)
            loss = self.loss_function(pred, value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        val_loss = 0.
        preds = []
        reals = []
        dataloader = dataModule.valid_loader if mode == 'valid' else dataModule.test_loader
        for batch in tqdm(dataloader):
            useridx, itemidx, value = batch
            useridx, itemidx, value = useridx.to(self.args.device), itemidx.to(self.args.device), value.to(self.args.device)
            pred = self.forward(useridx, itemidx)
            if mode == 'valid':
                val_loss += self.loss_function(pred, value)
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds.append(pred)
            reals.append(value)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        if mode == 'valid':
            self.scheduler.step(val_loss)
        metrics_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)

        # recall = recall_at_k(reals * dataModule.max_value, preds * dataModule.max_value, 10)
        # ndcg = ndcg_at_k(reals * dataModule.max_value, preds * dataModule.max_value, 10)
        # print(recall, ndcg)
        return metrics_error

def recall_at_k(reals, preds, k=10):
    recalls = []
    for real, pred in zip(reals, preds):
        top_k_preds = np.argsort(pred)[::-1][:k]
        hits = len(set(top_k_preds) & set(real))
        recalls.append(hits / len(real))
    return np.mean(recalls)

def ndcg_at_k(reals, preds, k=10):
    ndcgs = []
    for real, pred in zip(reals, preds):
        top_k_preds = np.argsort(pred)[::-1][:k]
        dcg = 0
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(real), k))])
        for i, p in enumerate(top_k_preds):
            if p in real:
                dcg += 1.0 / np.log2(i + 2)
        ndcgs.append(dcg / idcg)
    return np.mean(ndcgs)


def RunOnce(args, runId, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(datamodule, args)
    monitor = EarlyStopping(args)

    try:
        args.record = False
        model_path = f'./checkpoints/{log_filename}_{args.seed}.pt'
        model.load_state_dict(torch.load(model_path))
        results = model.evaluate_one_epoch(datamodule, 'test')
        if not args.classification:
            log.only_print(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f}')
        else:
            log.only_print(f'Acc={results["Acc"]:.4f} F1={results["F1"]:.4f} Precision={results["P"]:.4f} Recall={results["Recall"]:.4f}')
    except:
        # Setup training tool
        model.setup_optimizer(args)
        train_time = []
        for epoch in range(args.epochs):
            epoch_loss, time_cost = model.train_one_epoch(datamodule)
            valid_error = model.evaluate_one_epoch(datamodule, 'valid')
            monitor.track_one_epoch(epoch, model, valid_error)
            train_time.append(time_cost)
            log.show_epoch_error(runId, epoch, monitor, epoch_loss, valid_error, train_time)
            plotter.append_epochs(valid_error)
            if monitor.early_stop:
                break
        model.load_state_dict(monitor.best_model)
        sum_time = sum(train_time[: monitor.best_epoch])
        results = model.evaluate_one_epoch(datamodule, 'test')
        log.show_test_error(runId, monitor, results, sum_time)
        # Save the best model parameters
        makedir('./checkpoints')
        model_path = f'./checkpoints/{log_filename}_{args.seed}.pt'
        torch.save(monitor.best_model, model_path)
        log.only_print(f'Model parameters saved to {model_path}')
    return results


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        plotter.reset_round()
        results = RunOnce(args, runId, log)
        plotter.append_round()
        for key in results:
            metrics[key].append(results[key])
    log('*' * 20 + 'Experiment Results:' + '*' * 20)

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    if args.record:
        log.save_result(metrics)
        plotter.record_metric(metrics)

    log('*' * 20 + 'Experiment Success' + '*' * 20)

    return metrics


if __name__ == '__main__':
    args = get_config()
    set_settings(args)

    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model.upper()}, Density : {args.density}"
    log_filename = f'r{args.dimension}'
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log
    log(str(args.__dict__))

    # Run Experiment
    RunExperiments(log, args)


