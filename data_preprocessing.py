import os.path
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import torch
import pickle
import random
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# label 데이터가 전부 .LAB 확장자로 되어 있어서 .txt로 바꾸는 함수. 한번만 실행시키면 됨.
def LAB_to_txt(path):
    # :param path: 데이터셋 path. (본인 디렉토리 path)+/Data-EEG-25-users-Neuromarketing (str)
    files = glob.glob(path+'/labels/*.LAB')
    for name in files:
        if not os.path.isdir(name):
            src = os.path.splitext(name)
            os.rename(name, src[0]+'.txt')

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def BP_filter(data, lowcut, highcut, fs, order=5):
    # sos = butter_bandpass(lowcut, highcut, fs, order=order)
    # filtered = sosfilt(sos, data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


class Dataset(object):
    def __init__(self, subject, path, split=None, window_size=1, scaler='standard', batch_size=8, random_state=42):
        '''
        Input:
        :param path: 데이터셋 path. (본인 디렉토리 path)+/Data-EEG-25-users-Neuromarketing (str)
        :param split: train valid test 분리용 파라미터. Ex. [0.8, 0.1, 0.1] (list)
        :param window_size: 윈도우를 몇초로 할지 1, 2, 4초 실험해보셈  (int)
        :param scaler: StandardScaler - 'standard', MinMaxScaler - 'minmax', 안할거면 None
        :param batch_size: batch_size (int)
        :param random_state: seed (int)
        '''
        self.subject = subject
        self.feature_path = glob.glob(path+'/25-users/'+subject+'*')
        self.label_path = glob.glob(path+'/labels/'+subject+'*')
        self.window = window_size
        self.channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.sampling_rate = 128
        self.scaler = StandardScaler() if (scaler is not None) and (scaler == 'standard') else MinMaxScaler() if (scaler is not None) and (scaler == 'minmax') else None
        self.random_state = random_state
        self.split = split if split is not None else [0.8, 0.1, 0.1]
        self.batch_size = batch_size

    def read_data(self, feature_path, label_path):
        df = pd.read_csv(feature_path, sep=' ', header=None)
        op = open(label_path, 'r')
        label = op.read()
        op.close()
        return df, str(label)

    def data_to_df(self):
        data = None
        for i in tqdm(range(len(self.feature_path))):
            r_data, label = self.read_data(self.feature_path[i], self.label_path[i])
            r_data['label'] = 1 if str(label) == 'Like' else 0
            if i >= 1:
                data = pd.concat([data, r_data], axis=0)
            else:
                data = r_data
        y = data[['label']]
        X = data.drop('label', axis=1)
        for i in X.columns:
            X[i] = BP_filter(X[i].values, 4, 45, 128)
        return X, y

    def create_window(self, X_df, y_df):
        window_size = self.window * self.sampling_rate
        xs = []
        ys = []
        for i in tqdm(range(X_df.shape[0] // window_size)):
            x = X_df.iloc[(i * window_size):(i * window_size + window_size)].values
            y = y_df.iloc[i * window_size + window_size - 1].values
            xs.append(x.T)
            ys.append(y.T)
        return np.array(xs), np.array(ys)

    def make_Tensor(self, array):
        return torch.from_numpy(array).float()

    def return_data(self):
        X, y = self.data_to_df()
        if self.scaler:
            self.scaler.fit(X)
            X = pd.DataFrame(self.scaler.transform(X), columns=self.channels)
        X_win, y_win = self.create_window(X, y)
        print(X_win.shape, y_win.shape)
        print(self.subject, ": Data Preprocessing is Done.")
        return self.subject, (X_win, y_win)# {'train': train_tensor, 'valid': val_tensor, 'test': test_tensor}


if __name__=="__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    path = 'Write your dataset path'
    subject = ['Abhishek', 'Ankur_sir', 'Gautam', 'Gautam_123', 'Girvar_yadav', 'Kishore_babu', 'mahendra', 'Mohit', 'pawan_sahu', 'pradeep', 'Rajesh_el', 'rajkumar', 'Ravi_baba', 'Ravi_ph', 'Rockysingh', 'Rupak', 'Sachin', 'Sandeep', 'Soumendu', 'Suraj_sir', 'taufiq', 'Veerpal', 'Vijay', 'Vipin_1', 'Viraj_1']
    a = {}
    for s in subject:
        sub, data = Dataset(subject=s, path=path, window_size=4).return_data()
        a[str(sub)] = data
    with open('./dataset_sd.pkl', 'wb') as f:
        pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)