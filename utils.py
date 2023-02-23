import copy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.manifold import TSNE
import mydata
if __name__ == '__main__':
  print(Pan_data_path)

def get_trainpath(source_path):
  source_data_path,source_train_set,source_test_set=[],[],[]
  if source_path == 'temp':
      source_data_path = mydata.temp_data_path
      source_train_set = mydata.temp_train_set
      source_test_set = mydata.temp_test_set
  elif source_path == 'Pan':
      source_data_path = mydata.Pan_data_path
      source_train_set = mydata.Pan_train_set
      source_test_set = mydata.Pan_test_set
  elif source_path == 'LG':
      source_data_path = mydata.LG_data_path
      source_train_set = mydata.LG_train_set
      source_test_set = mydata.LG_test_set
  elif source_path == 'INR':
      source_data_path = mydata.INR_data_path
      source_train_set = mydata.INR_train_set
      source_test_set = mydata.INR_test_set
  return source_data_path,source_train_set,source_test_set

def get_testpath(target_path):
  target_data_path,target_train_set,target_test_set=[],[],[]
  if target_path == 'temp':
      target_data_path = mydata.temp_data_path
      target_train_set = mydata.temp_train_set
      target_test_set = mydata.temp_test_set
  elif target_path == 'pseudo':
      target_data_path = mydata.pseudo_data_path
      target_train_set = mydata.pseudo_train_set
      target_test_set = mydata.pseudo_test_set
  elif target_path == 'Pan':
      target_data_path = mydata.Pan_data_path
      target_train_set = mydata.Pan_train_set
      target_test_set = mydata.Pan_test_set
  elif target_path == 'LG':
      target_data_path = mydata.LG_data_path
      target_train_set = mydata.LG_train_set
      target_test_set = mydata.LG_test_set
  elif target_path == 'INR':
      target_data_path = mydata.INR_data_path
      target_train_set = mydata.INR_train_set
      target_test_set = mydata.INR_test_set
  return target_data_path,target_train_set,target_test_set


def plot_tsne(features,labels,name):
  ts = TSNE(n_components=3,init='pca',random_state=0)
  res = ts.fit_transform(features)
  fig = plt.figure()
  for i in range(features.shape[0]):
    plt.scatter(res[i,0],res[i,1],color=plt.cm.Set1(labels[i]+3),marker='*',s=5)
  plt.savefig(name)
  plt.close()

def new_plot_data(X, Y):
  time = list(range(X.shape[0]))
  voltage = X[:, 0].T
  current = X[:, 1].T
  battery_temp = X[:, 2].T
  ah = Y.T.squeeze()
  name = 'plot'
  fig, axes = plt.subplots(4, figsize = (32,16))
  fig.subplots_adjust(hspace = 0.5)
  suptitle = name

  axes[0].plot(time, voltage)
  axes[0].set_xlabel('Time(s)')
  axes[0].set_ylabel('Voltage(V)')
  axes[0].set_yticks(np.arange(0, 1, 0.1))
  axes[0].grid(True, linestyle = '-')

  axes[1].plot(time, current)
  axes[1].set_xlabel('Time(s)')
  axes[1].set_ylabel('Current(A)')
  axes[1].set_yticks(np.arange(0, 1, 0.1))
  axes[1].grid(True, linestyle = '-')

  axes[3].plot(time, battery_temp)
  axes[3].set_xlabel('Time(s)')
  axes[3].set_ylabel('Battery_Temp_degC(℃)')
  axes[3].set_yticks(np.arange(0, 1, 0.1))
  axes[3].grid(True, linestyle = '-')

  axes[2].plot(time, ah)
  axes[2].set_xlabel('Time(s)')
  axes[2].set_ylabel('Capacity(Ah)')
  axes[2].set_yticks(np.arange(0, 1, 0.1))
  axes[2].grid(True, linestyle = '-')

  plt.show()
  plt.close()

def plot_data(time, voltage, current, battery_temp, ah, train_name):
    fig, axes = plt.subplots(2, 2, figsize = (16,8))
    fig.subplots_adjust(hspace = 0.5)
    suptitle = train_name

    axes[0,0].plot(time, voltage)
    axes[0,0].set_xlabel('Time(s)')
    axes[0,0].set_ylabel('Voltage(V)')
    axes[0,0].set_yticks(np.arange(2.3, 4.5, 0.2))
    axes[0,0].grid(True, linestyle = '-')

    axes[0,1].plot(time, current)
    axes[0,1].set_xlabel('Time(s)')
    axes[0,1].set_ylabel('Current(A)')
    axes[0,1].set_yticks(np.arange(-20, 15, 3))
    axes[0,1].grid(True, linestyle = '-')

    axes[1,0].plot(time, ah)
    axes[1,0].set_xlabel('Time(s)')
    axes[1,0].set_ylabel('Capacity(Ah)')
    axes[1,0].set_yticks(np.arange(-2.7, 0, 0.3))
    axes[1,0].grid(True, linestyle = '-')
    axes[1,1].plot(time, battery_temp)
    axes[1,1].set_xlabel('Time(s)')
    axes[1,1].set_ylabel('Battery_Temp_degC(℃)')
    axes[1,1].grid(True, linestyle = '-')
    plt.show()
    plt.close()

def plot_result(rundir,Y_test, Y_predict, save_image, test_name):
  Y_test = Y_test.cpu().detach().numpy()
  Y_predict = Y_predict.cpu().detach().numpy()
  
  Y_test = Y_test.flatten()
  Y_predict = Y_predict.flatten()
  plt.figure(figsize = (18,5))
  plt.title(test_name)
  plt.plot(Y_test, label = 'Y_test')
  plt.plot(Y_predict, label = 'predict')
  plt.xlabel('Time(s)')
  plt.ylabel('Capacity(Ah)')
  plt.yticks(np.arange(0.0, 1.1, 0.1))
  plt.grid(True,linestyle = '-')
  plt.legend()
  path = rundir + '/images/'+ save_image + '/' + test_name + '.jpg'
  plt.savefig(path)
  plt.show()
  plt.close()

def save_error(rundir,data, test_name, save_error='test'):
    filename = rundir + '/errors/' + save_error + '/' + test_name
    file = open(filename, 'w')
    file.write(data + '\n')
    file.close()

def save_min(rundir,min_mae,min_rmse,min_max,epoch,domain_loss=0,domain_acc=0):
      filename = rundir + '/errors/min_errors'
      f = open(filename,'w')
      f.write('mae' + str(min_mae) + '\n')
      f.write('rmse' + str(min_rmse) + '\n')
      f.write('max' + str(min_max) + '\n')
      f.write('epoch ' + str(epoch) + '\n')
      f.write('domain_loss ' + str(domain_loss) + '\n')
      f.write('domain_acc ' + str(domain_acc) + '\n')
      f.close()

def plot_train_loss(rundir,loss_train_domain, loss_train_predictor, domain_accuracy, epoch):
  plt.figure(figsize = (16,16))
  name='loss iteration'
  plt.title(name)
  plt.plot(loss_train_domain, label='loss_domain')
  plt.plot(loss_train_predictor, label='loss_predictor')
  plt.plot(domain_accuracy, label='domain_accuracy')
  plt.xlabel('epoch')
  plt.ylabel('mse loss/accuracy')
  plt.grid(True, linestyle='-')
  plt.legend()
  path = rundir+'/loss_iter/train_loss/epoch_' + str(epoch) +  '.jpg'
  plt.savefig(path)
  plt.show()
  plt.close()

def plot_test_loss(rundir,loss_mae, loss_rmse, loss_max, epoch):
  plt.figure(figsize = (16,16))
  plt.title('test loss')
  plt.plot(loss_rmse, label = 'loss_rmse')
  plt.plot(loss_mae, label = 'loss_mae')
  plt.plot(loss_max, label = 'loss_max')
  plt.xlabel('epoch')
  plt.ylabel('test loss')
  #plt.yticks(np.arange(0.0, 1.1, 0.1))
  plt.grid(True,linestyle = '-')
  path = rundir + '/loss_iter/test_loss/epoch' + str(epoch) + '.jpg'
  plt.legend()
  plt.savefig(path)
  plt.show()
  plt.close()

def MAXLoss(y_predict, y_label):
  return (torch.max(torch.abs(y_predict - y_label))).item()

def mkdir(dir):
  if dir == None:
    i = 1
    while True:
        rundir = './run/' +str(i)
        i += 1
        if not os.path.exists(rundir):
          break
  else:
    rundir = './run/' + dir
    if os.path.exists(rundir):
      i = 1
      while True:
        rundir = './run/' + dir + '(' + str(i) + ')'
        i += 1
        if not os.path.exists(rundir):
          break
  os.makedirs(rundir)
  os.makedirs(rundir+'/errors/test')
  os.makedirs(rundir+'/errors/train')
  os.makedirs(rundir+'/images/test')
  os.makedirs(rundir+'/images/train')
  os.makedirs(rundir+'/loss_iter/test_loss')
  os.makedirs(rundir+'/loss_iter/train_loss')
  os.makedirs(rundir+'/saved_model')
  os.makedirs(rundir+'/tsne')
  return rundir