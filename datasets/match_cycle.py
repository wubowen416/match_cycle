import os
import pickle
import numpy as np
import torch as th
import joblib as jl

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MatchCycleDatabase:
    def __init__(
        self, 
        dataset_path: str,
        y_maximum: float,
        test_size: float,
        seed: int
    ):
        cached_data_path = os.path.join(
            os.path.dirname(dataset_path), 
            "data_cached_ymax_{}_tsize_{}_seed_{}.bin".format(y_maximum, test_size, seed)
        )
        if os.path.exists(cached_data_path):
            with open(cached_data_path, "rb") as f:
                data = pickle.load(f)
            Xs = data['Xs']
            ys = data['ys']
        else:
            # load from txt
            with open(dataset_path, "r") as f:
                lines = f.readlines()
            # read & save data
            Xs, ys = [], []
            for line in lines:
                elements = line.split(' ')
                y = float(elements[-1].strip('\n'))
                # filter out larger than max_value
                if y > y_maximum:
                    continue
                Xs.append([float(e) for e in elements[:5]])
                ys.append(y)
            # to ndarray
            Xs = np.array(Xs)
            ys = np.array(ys)[:, np.newaxis] # (N,) to (N, 1)
            with open(cached_data_path, "wb") as f:
                pickle.dump(dict(Xs=Xs, ys=ys), f)
        # split to train, val & test
        Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=test_size)
        Xs_train, Xs_val, ys_train, ys_val = train_test_split(Xs_train, ys_train, test_size=test_size)
        # init scalers
        X_scaler_path = os.path.join(
            os.path.dirname(dataset_path),
            "X_scaler_ymax_{}_tsize_{}_seed_{}.bin".format(y_maximum, test_size, seed)
        )
        if os.path.exists(X_scaler_path):
            X_scaler = jl.load(X_scaler_path)
        else:
            X_scaler = StandardScaler().fit(Xs_train)
            jl.dump(X_scaler, X_scaler_path)
        y_scaler_path = os.path.join(
            os.path.dirname(dataset_path),
            "y_scaler_ymax_{}_tsize_{}_seed_{}.bin".format(y_maximum, test_size, seed)
        )
        if os.path.exists(y_scaler_path):
            y_scaler = jl.load(y_scaler_path)
        else:
            y_scaler = StandardScaler().fit(ys_train)
            jl.dump(y_scaler, y_scaler_path)
        # scale data
        Xs_train = X_scaler.transform(Xs_train)
        ys_train = y_scaler.transform(ys_train)
        Xs_val = X_scaler.transform(Xs_val)
        ys_val = y_scaler.transform(ys_val)
        Xs_test = X_scaler.transform(Xs_test)
        ys_test = y_scaler.transform(ys_test)
        # init dataset
        self.train_dataset = MyDataset(Xs_train, ys_train)
        self.val_datset = MyDataset(Xs_val, ys_val)
        self.test_dataset = MyDataset(Xs_test, ys_test)
        self.X_dim = X_scaler.mean_.shape[0]
        self.y_dim = y_scaler.mean_.shape[0]
        self.X_mean = X_scaler.mean_
        self.X_scale = X_scaler.scale_
        self.y_mean = y_scaler.mean_
        self.y_scale = y_scaler.scale_

    def get_dims(self):
        return dict(
            d_in=self.X_dim, 
            d_out=self.y_dim,
            X_mean=self.X_mean,
            X_scale=self.y_scale,
            y_mean=self.y_mean,
            y_scale=self.y_scale
        )
    
    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_datset

    def get_test_dataset(self):
        return self.test_dataset


class MyDataset(Dataset):
    def __init__(self, Xs, ys):
        self.Xs = Xs
        self.ys = ys
    
    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        X = th.from_numpy(self.Xs[idx]).float()
        y = th.from_numpy(self.ys[idx]).float()
        return X, y