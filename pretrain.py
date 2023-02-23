from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import argparse
from tqdm import tqdm
import math
from utils import *
from mydata import *
from models import *
eval_interval = 100
batch_size = 6

def pretrain(rundir,source_temp,target_temp,source_data_path,source_train_set,source_test_set,models, criterion, optimizers, batch_size, epochs,eval_interval, seed=0, device_type=('cuda:0' if torch.cuda.is_available() else 'cpu'), ifsave=True, load_model=False, load_model_path='./saved_model/best.pt'):
  loss_min = 10000
  rundir = mkdir(rundir)
  device = torch.device(device_type)
  if load_model:
      load_saved_model(device,models,optimizers,loss_min,seed)
  if torch.cuda.is_available():
    for  model in models:
      models[model].to(device)
  
  init_seed(seed)
  criterion_mae = nn.L1Loss()
  criterion_mse = nn.MSELoss()
  for temp_idx in range(1):
    source_data = Mydataset(source_data_path, source_temp, source_train_set,mode='train')
    source_loader = DataLoader(source_data, batch_size=batch_size, shuffle=True)
    source_test_data = Mydataset(source_data_path, source_temp, source_test_set,mode='test')
    source_test_loader = DataLoader(source_test_data, batch_size=1, shuffle=False)

    loss_iter_domain = []
    loss_iter_predictor = []
    loss_iter_test = []
    loss_iter_mae = []
    loss_iter_rmse = []
    loss_iter_max = []
    loss_iter_domain_acc = []
    loss_train_predictor = 0
    test_len = len(source_test_loader)
    min_max,min_mae,min_rmse = [],[],[]
    for i in range(test_len):
      min_mae.append(1)
      min_rmse.append(1)
      min_max.append(1)
    #checkpoint = torch.load(load_model_path, map_location=device)
     #models['domain_classifier'].load_state_dict(checkpoint['domain_classifier'])
    for epoch in range(epochs+500):
      ##########
      #train
      ##########
      for model in models:
        models[model].train()
      loss_train = 0
      loss_test = 0
      source_sample = 0
      #tqdm_mix = tqdm(source_loader,desc='epoch '+str(epoch))
      for i, (source_data, source_label) in enumerate(source_loader):
        source_data = source_data.to(device)
        source_label = source_label.to(device)

        for op in optimizers:
          optimizers[op].zero_grad()

        source_features = models['conv'](source_data)
        predict_label = models['regression']\
                          (models['fc']\
                            (models['lstm'](source_features))).squeeze()
        predict_loss = criterion(predict_label,source_label)
        loss = predict_loss
        loss.backward()
        for op in optimizers:
          optimizers[op].step()
        loss_train += loss.item()
        source_sample += len(source_data)
        #if ((epoch+1) % eval_interval) == 0:
        #  plot_result(source_label, predict_label, save_image='train', 
        #          test_name=source_data_path[7:-1] + source_temp + '_epoch_' + str(epoch))
      loss_train = loss_train/(source_sample)
      print('epoch {}:loss {}'.format(epoch, loss_train))
      if (loss_train < loss_min) & (ifsave==True):
        loss_min = loss_train
        path = rundir+'/saved_model/best.pt'
        #save_model(models, optimizers, loss_min, seed,path)
        #print('min loss:{} saved model'.format(loss_min))
      ##########
      #test
      ##########
      for model in models:
        models[model].eval()
      
      #tqdm_test = tqdm(source_test_loader, desc='source data test')
      loss_mae = 0
      loss_rmse = 0
      loss_max = 0
      if ((epoch+1) % eval_interval) == 0:
            print('source test res')
      for i, data in enumerate(source_test_loader):
        x_test, y_test = data
        x_test, y_test = x_test.squeeze(), y_test.squeeze()
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        with torch.no_grad():
          y_predict = models['regression']\
                            (models['fc']\
                              (models['lstm']\
                                (models['conv'](x_test)))).squeeze()

        loss_mse = criterion_mse(y_predict, y_test)
        loss_test = loss_mse.detach().cpu().item()
        loss_mae = criterion_mae(y_predict, y_test).detach().cpu().item()
        loss_rmse = math.sqrt(loss_mse.detach().cpu().item())
        loss_max = MAXLoss(y_predict, y_test)
        if epoch > epochs:  
          min_avg = (min_mae[i] + min_rmse[i])/2
          loss_avg = (loss_mae + loss_rmse) / 2
          if min_avg > loss_avg:
            min_max[i] = loss_max
            min_rmse[i] = loss_rmse
            min_mae[i] = loss_mae
            test_name=source_data_path[-4:-1] + source_temp + source_test_set[i]+'best'
            plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
            error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
            print(error)
            save_error(rundir,error,test_name,'test')
            loss_min = loss_train
            path = rundir+'/saved_model/best.pt'
            save_model(models, optimizers, loss_min, seed,path)
            print('min avg loss:{} saved model'.format(loss_avg))

        save_min(rundir,min_mae,min_rmse,min_max,epoch)
        if ((epoch+1) % eval_interval) == 0:
          test_name=source_data_path[-4:-1] + source_temp + source_test_set[i]
          plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
          error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
          print(error)
          save_error(rundir,error,test_name,'test')
      #####
      loss_iter_domain.append(loss_train)
      loss_iter_predictor.append(loss_train_predictor)
      loss_iter_test.append(loss_test)
      loss_iter_mae.append(loss_mae)
      loss_iter_rmse.append(loss_rmse)
      loss_iter_max.append(loss_max)
      #if ( ((epoch+1) % eval_interval)==0 ) & (ifsave==True):
      #  save_model(models, optimizers, loss_min, seed, model_path='./saved_model/epoch'+str(epoch)+'.pt')
    plot_train_loss(rundir,loss_iter_domain, loss_iter_predictor, loss_iter_domain_acc, epochs)
    plot_test_loss(rundir,loss_iter_mae, loss_iter_rmse, loss_iter_max, epochs)
    
