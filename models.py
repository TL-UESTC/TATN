import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.functional import relu
import numpy as np
import random

filters = 256
kernel_size = 3
dropout = 0.2
hidden_units = 64

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def mmd(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()
  
def init_seed(seed=0):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

class conv(nn.Module):
  def __init__(self):
    super(conv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(1, filters,(3,1),padding='same'),#input: N *1* 3 * 90  output: N * 20 *3* 90
        nn.ReLU(),
        nn.Dropout(dropout),
        
        nn.Conv2d(filters,filters,(3,1),padding='same'),
        nn.ReLU(),
        nn.Dropout(dropout)
    )
    
  def forward(self, x):
    if type(x)==np.ndarray:
      x = torch.from_numpy(x)
    x = x.reshape(-1,1,3,90)
    x = self.conv(x)
    x = x.reshape(-1,filters*3,90)
    return x

class lstm(nn.Module):
  def __init__(self):
    super(lstm, self).__init__()
    self.lstm1 = nn.Sequential(
        nn.LSTM(filters*3, hidden_units, 2, bidirectional=True, batch_first=True)#input: N * 90 * 256 output:N * 90 * 128
    )
    self.lstm2 = nn.Sequential(
        nn.LSTM(hidden_units*2, hidden_units, 2, bidirectional=True, batch_first=True),
    )

  def forward(self, x):
    if type(x)==np.ndarray:
      x = torch.from_numpy(x)
    x = x.permute(0, 2, 1)
    x, (h,c) = self.lstm1(x)
    x, (h,c) = self.lstm2(x)
    return x

class fc(nn.Module):
  def __init__(self):
    super(fc, self).__init__()
    self.fc = nn.Sequential(
        nn.Linear(2*hidden_units, 2*hidden_units),
        nn.ReLU()
    )

  def forward(self, x):
    if type(x)==np.ndarray:
      x = torch.from_numpy(x)
    x = self.fc(x)  
    return x

class regression(nn.Module):
  def __init__(self):
    super(regression, self).__init__()
    self.fc = nn.Linear(2*hidden_units, 1)
    
  def forward(self, x):
    if type(x)==np.ndarray:
      x = torch.from_numpy(x)
    x = self.fc(x)
    #x = sigmoid(x)
    return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.lstm1 = nn.Sequential(
          nn.LSTM(filters*3, hidden_units, 2, bidirectional=True, batch_first=True)#input: N * 90 * 256 output:N * 90 * 128
        )
        self.lstm2 = nn.Sequential(
          nn.LSTM(hidden_units*2, hidden_units, 2, bidirectional=True, batch_first=True),
        )
        self.fc1 = nn.Sequential(
          nn.Linear(3*filters, 3*filters),
          nn.ReLU()
        )
        self.fc2 = nn.Sequential(
          nn.Linear(2*hidden_units*90,1),
          nn.Sigmoid()
        )
  
    def forward(self,x):
      if type(x)==np.ndarray:
        x = torch.from_numpy(x)
      #x = x.permute(0, 2, 1)
      #x, (h,c) = self.lstm1(x)
      #x, (h,c) = self.lstm2(x)
      #x = self.fc1(x)
      x = torch.reshape(x,(x.shape[0],-1))
      x = self.fc2(x)
      return x

class gan_generator(nn.Module):
    def __init__(self):
      super(gan_generator,self).__init__()
      pass

def save_model(models, optimizers, loss_min, seed, model_path='./saved_model/best.pt'):
  torch.save({
        'conv':models['conv'].state_dict(),
        'lstm':models['lstm'].state_dict(),
        'fc':models['fc'].state_dict(),
        'regression':models['regression'].state_dict(),
        'optimizer_conv':optimizers['conv'].state_dict(),
        'optimizer_lstm':optimizers['lstm'].state_dict(),
        'optimizer_fc':optimizers['fc'].state_dict(),
        'optimizer_regression':optimizers['regression'].state_dict(),
        'loss_min':loss_min,
        'seed':seed
        }, model_path )

def load_saved_model(device,models,optimizers,loss_min,seed,model_path='./saved_model/best.pt'):
    device = torch.device('cuda')
    ckpt = torch.load(model_path,map_location=device)
    layers=['conv','lstm','fc','regression']
    for l in layers:
        models[l+'_s'].load_state_dict(ckpt[l])
        models[l].load_state_dict(ckpt[l])
        #optimizers[l].load_state_dict(ckpt['optimizer_'+l])
    loss_min = ckpt['loss_min']
    seed = ckpt['seed']

