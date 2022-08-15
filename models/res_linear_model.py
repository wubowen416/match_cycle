import os
import tqdm
import wandb
os.environ['WANDB_MODE'] = "online"
import numpy as np
import torch as th
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from .nets import ResLinearNet


class ResLinearModel:
    def __init__(
        self,
        args,
        chkpt_path: str = "",
        device: str = ""
    ):
        # modify args according to provided arguments
        if chkpt_path != "":
            args.chkpt_path = chkpt_path
        if device != "":
            args.device = device
        self.args = args
        # init model
        self.best_chkpt_path = os.path.join(
            os.path.join(self.args.chkpt_path, self.args.run_name), 
            "best.pt"
        )
        os.makedirs(os.path.dirname(self.best_chkpt_path), exist_ok=True)
        self.last_chkpt_path = self.best_chkpt_path.replace("best", "last")
        self.initialize_net()
        self.net.to(args.device)
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.early_stopping_counter = 0
        if os.path.exists(self.best_chkpt_path):
            self.load_best()
            wandb.init(project=self.args.project, id=self.run_id, resume="must")
        else:
            print("Model: initialize new model.")
            self.run_id = wandb.util.generate_id()
            wandb.init(project=args.project, name=args.run_name, id=self.run_id, config=args)
            self.epoch = 0
            self.train_step = 0
            self.best_loss = 99999
            self.checkpoint(99999-1)
        # define metrics
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/epoch")
        wandb.define_metric("val/*", step_metric="val/epoch")
        wandb.define_metric("val/loss", summary="min", step_metric="val/epoch")

    def initialize_net(self):
        self.net = ResLinearNet(
            d_in=self.args.d_in,
            d_out=self.args.d_out,
            d_model=self.args.d_model,
            num_layers=self.args.num_layers
        )

    def forward(self, Xs):
        return self.net(Xs)

    def initialize_optimizer(self):
        self.optimizer = th.optim.Adam(
            self.net.parameters(),
            betas=(self.args.momentum, self.args.adagrad),
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay
        )

    def initialize_scheduler(self):
        self.scheduler = th.optim.lr_scheduler.StepLR(
            self.optimizer, self.args.scheduler_step_size
        )

    def load_best(self):
        chkpt = th.load(self.best_chkpt_path, map_location=self.args.device)
        print("Model: Load best from {}.".format(self.best_chkpt_path))
        self._load_chkpt(chkpt)

    def load_last(self):
        assert os.path.exists(self.last_chkpt_path), "{} does not exist.".format(self.last_chkpt_path)
        chkpt = th.load(self.last_chkpt_path, map_location=self.args.device)
        print("Model: Load last from {}.".format(self.last_chkpt_path))
        self._load_chkpt(chkpt)

    def _load_chkpt(self, chkpt):
        self.net.load_state_dict(chkpt['net'])
        self.optimizer.load_state_dict(chkpt['optimizer'])
        self.scheduler.load_state_dict(chkpt['scheduler'])
        self.epoch = chkpt['epoch']
        self.train_step = chkpt['train_step']
        self.best_loss = chkpt['best_loss']
        self.run_id = chkpt['run_id']

    def checkpoint(self, val_loss: float) -> bool:
        chkpt = dict(
            net = self.net.state_dict(),
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            args = self.args,
            epoch = self.epoch,
            train_step = self.train_step,
            best_loss = self.best_loss,
            run_id = self.run_id
        )
        th.save(chkpt, self.last_chkpt_path)
        if val_loss < self.best_loss:
            print("val loss decreased from {} to {}. Model saved at {}".format(self.best_loss, val_loss, self.best_chkpt_path))
            self.best_loss = val_loss
            self.early_stopping_counter = 0
            th.save(chkpt, self.best_chkpt_path)
        else:
            print("val loss did not decreased. Earlystopping count: {} out of {}.".format(self.early_stopping_counter, self.args.early_stopping))
            self.early_stopping_counter += 1
        if self.early_stopping_counter == self.args.early_stopping:
            print("Earlystopping threshold encounterd. Training stopped.")
            return True
        else:
            return False

    def get_data_loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers
        )

    def compute_loss(self, y_true, y_pred):
        mse = F.mse_loss(y_pred, y_true)
        l1 = F.l1_loss(y_pred, y_true)
        ratio = self.ratio_wrt_target_rescaled(y_true, y_pred)
        total_loss = self.args.lambda_mse * mse
        total_loss += self.args.lambda_l1 * l1
        total_loss += self.args.lambda_ratio * ratio
        return dict(
            loss = total_loss,
            mse = mse,
            l1 = l1,
            ratio = ratio
        )

    def ratio_wrt_target_rescaled(self, y_true, y_pred):
        y_mean = th.from_numpy(self.args.y_mean).float().to(self.args.device)
        y_scale = th.from_numpy(self.args.y_scale).float().to(self.args.device)
        y_true = y_scale * y_true + y_mean
        y_pred = y_scale * y_pred + y_mean
        ratio = th.abs(y_true - y_pred) / th.abs(y_true)
        return th.mean(ratio)

    def fit(
        self, 
        train_dataset: Dataset, 
        val_dataset: Dataset
    ) -> None:
        if os.path.exists(self.last_chkpt_path):
            self.load_last()
        train_loader = self.get_data_loader(train_dataset, shuffle=True)
        val_loader = self.get_data_loader(val_dataset)
        for _ in range(self.args.n_epochs):
            self.epoch += 1
            print("Epoch: {}".format(self.epoch))
            # train
            self.net.train()
            for Xs, ys_true in tqdm.tqdm(train_loader, ascii=True):
                self.train_step += 1
                Xs = Xs.to(self.args.device)
                ys_true = ys_true.to(self.args.device)
                ys = self.forward(Xs)
                loss_terms = self.compute_loss(ys_true, ys)
                self.optimizer.zero_grad()
                loss_terms['loss'].backward()
                self.optimizer.step()
                log_dict = {
                    "train/step": self.train_step,
                    "train/grad_norm": self.grad_norm(self.net)
                }
                for loss_name, value in loss_terms.items():
                    log_dict[f"train/{loss_name}"] = value.item()
                wandb.log(log_dict)
            # val
            log_dict = {"val/epoch": self.epoch}
            for loss_name in loss_terms.keys():
                log_dict[f"val/{loss_name}"] = 0
            self.net.eval()
            for Xs, ys_true in val_loader:
                Xs = Xs.to(self.args.device)
                ys_true = ys_true.to(self.args.device)
                with th.no_grad():
                    ys = self.forward(Xs)
                loss_terms = self.compute_loss(ys_true, ys)
                for loss_name, value in loss_terms.items():
                    log_dict[f"val/{loss_name}"] += value.item() / len(val_loader)
            wandb.log(log_dict)
            early_stop = self.checkpoint(log_dict['val/loss'])
            if early_stop:
                break

    def eval(self, dataset: Dataset):
        print("Evaluate model.")
        data_loader = self.get_data_loader(dataset)
        self.net.eval()
        ys_true_all, ys_all = [], []
        for Xs, ys_true in tqdm.tqdm(data_loader, ascii=True):
            Xs = Xs.to(self.args.device)
            ys_true = ys_true.to(self.args.device)
            with th.no_grad():
                ys = self.forward(Xs)
            ys_true_all.append(ys_true)
            ys_all.append(ys)
        ys_true_all = th.cat(ys_true_all, dim=0)
        ys_all = th.cat(ys_all, dim=0)
        loss_terms = self.compute_loss(ys_true_all, ys_all)
        results = {}
        for loss_name, value in loss_terms.items():
            results[f"test/{loss_name}"] = value.item()
        wandb.log(results)

    @staticmethod
    def grad_norm(net: th.nn.Module) -> float:
        total_norm = 0
        for p in net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

    