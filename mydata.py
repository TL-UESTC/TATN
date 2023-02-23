import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from utils import *

steps = 90
feature_num = 3
pseudo_data_path = './normalized_data/Pan/'
Pan_data_path = './normalized_data/Pan/'
INR_data_path = './normalized_data/INR/'
LG_data_path = './normalized_data/LG/'
pseudo_train_set = []
pseudo_test_set = []
Pan_train_set = []
Pan_test_set = []
LG_train_set = []
LG_test_set = []

train_name = ['Cycle_1', 'Cycle_2', 'Cycle_3', 'Cycle_4', 'NN']
test_name = ['US06', 'HWFET', 'LA92', 'UDDS']
temp = ['25', '10', '0', 'n10', 'n20']
source_temp = ['25', '10', '0', 'n10', 'n20']
target_temp=['25', '10', '0', 'n10', 'n20']
class Mydataset(Dataset):
  def __init__(self, data_path,temp, set, mode='train'):
    self.X, self.Y = [], []
    total = 0
    for name in set:
      path = data_path + temp + '/' + temp + name

      mat = sio.loadmat(path)
      current,voltage,battery_temp,ah = mat['current'],mat['voltage'],mat['temp'],mat['ah']
      ah = ah[:len(ah)//steps*steps]
    
      data = np.c_[current,voltage,battery_temp]
      data = data[:len(data)//steps*steps]
      self.X.append(data)
      self.Y.append(ah)
    if mode == 'train':
      self.X,self.Y = np.concatenate(self.X,axis=0),np.concatenate(self.Y,axis=0)
      #print('shape:{},{}'.format(self.X.shape,self.Y.shape))

      new_plot_data(self.X, self.Y)
    
      self.X = np.reshape(self.X, (self.X.shape[0]//steps, feature_num, steps))
      self.Y = np.reshape(self.Y, (self.Y.shape[0]//steps, steps))
      #print('shape:{},{}'.format(self.X.shape,self.Y.shape))
      self.X = torch.from_numpy(self.X)
      self.Y = torch.from_numpy(self.Y)
      self.X = self.X.type(torch.FloatTensor)
      self.Y = self.Y.type(torch.FloatTensor)
    elif mode == 'test':
      for i in range(len(set)):
        #print('shape:{},{}'.format(self.X[i].shape,self.Y[i].shape))

        self.X[i] = np.reshape(self.X[i], (self.X[i].shape[0]//steps, feature_num, steps))
        self.Y[i] = np.reshape(self.Y[i], (self.Y[i].shape[0]//steps, steps))
        #print('shape:{},{}'.format(self.X[i].shape,self.Y[i].shape))
        self.X[i] = torch.from_numpy(self.X[i])
        self.Y[i] = torch.from_numpy(self.Y[i])
        self.X[i] = self.X[i].type(torch.FloatTensor)
        self.Y[i] = self.Y[i].type(torch.FloatTensor)
  
  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]
  
  def __len__(self):
    return len(self.X)

