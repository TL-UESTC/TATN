from re import T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from torch.autograd import Variable
import argparse
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")   
import numpy as np
from utils import *
from mydata import *
from models import *
eval_interval = 100
batch_size = 6
decay_iter = 50
decay_lamb = 1

def train(rundir,source_temp,target_temp,source_data_path,source_train_set,source_test_set,target_data_path,target_train_set,target_test_set,models, criterion, optimizers, batch_size, epochs,eval_interval, lamb1,lamb2,lamb3,seed=0, device_type=('cuda:0' if torch.cuda.is_available() else 'cpu'), ifsave=True, load_model=False, model_path='/models/best.pt'):
  loss_min = 10000
  rundir = mkdir(rundir)
  device = torch.device(device_type)
  if load_model:
      load_saved_model(device,models,optimizers,loss_min,seed,model_path)
  if torch.cuda.is_available():
    for  model in models:
      models[model].to(device)
    
  init_seed(seed)
  criterion = criterion
  criterion_mae = nn.L1Loss()
  criterion_mse = nn.MSELoss()
  domain_criterion = nn.BCELoss()
  for temp_idx in range(1):
    source_data = Mydataset(source_data_path, source_temp, source_train_set,mode='train')
    source_loader = DataLoader(source_data, batch_size=batch_size, shuffle=True)
    target_data = Mydataset(target_data_path, target_temp, target_train_set,mode='train')
    target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=True)
    source_test_data = Mydataset(source_data_path, source_temp, source_test_set,mode='test')
    source_test_loader = DataLoader(source_test_data, batch_size=1, shuffle=False)
    target_test_data_2 = Mydataset(target_data_path, target_temp, target_train_set, mode='test')
    target_test_loader_2 = DataLoader(target_test_data_2, batch_size=1, shuffle=False)
    target_test_data = Mydataset(target_data_path, target_temp, target_test_set, mode='test')
    target_test_loader = DataLoader(target_test_data, batch_size=1, shuffle=False)
    
    total_steps = epochs*len(source_loader)
    
    loss_iter_domain = []
    loss_iter_predictor = []
    loss_iter_test = []
    loss_iter_mae = []
    loss_iter_rmse = []
    loss_iter_max = []
    loss_iter_domain_acc = []
    test_len = len(target_test_loader)
    min_max,min_mae,min_rmse = [],[],[]
    for i in range(test_len):
      min_mae.append(1)
      min_rmse.append(1)
      min_max.append(1)
    #checkpoint = torch.load(load_model_path, map_location=device)
     #models['domain_classifier'].load_state_dict(checkpoint['domain_classifier'])
    for epoch in range(epochs):
      if ((epoch+1) % decay_iter) == 0:
        lamb3 = lamb3 * decay_lamb
      ##########
      #train
      ##########
      start_step = epoch*len(source_loader)
      for model in models:
        models[model].train()
      loss_train = 0
      loss_domain = 0
      loss_target_domain = 0
      loss_test = 0
      total_num = 0
      total_hit = 0
      tqdm_mix = tqdm(zip(source_loader, target_loader),desc='epoch '+str(epoch))
      source_sample = 0
      target_sample = 0
      domain_loss = 0
      domain_acc = 0
      for i, ((source_data, source_label), (target_data, target_label)) in enumerate(tqdm_mix):
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        target_label = target_label.to(device)
        mix_label = torch.zeros([source_data.shape[0]+target_data.shape[0],1]).to(device)
        mix_label[:source_data.shape[0]] = 1
        S_labels = torch.ones([source_data.shape[0],1]).to(device)
        T_labels = torch.zeros([target_data.shape[0],1]).to(device)
        
        p = float(i+start_step)/total_steps
        cons = 2 / (1 + np.exp(-10 * p)) - 1
        for op in optimizers:
          optimizers[op].zero_grad()
        #train domain discriminator
        source_features = models['lstm'](models['conv_s'](source_data))
        target_features = models['lstm'](models['conv'](target_data))
        source_domain_pred = models['discriminator'](source_features.detach())
        target_domain_pred = models['discriminator'](target_features.detach())
        source_domain_loss = domain_criterion(source_domain_pred,S_labels)
        target_domain_loss = domain_criterion(target_domain_pred,T_labels)
        domain_loss = source_domain_loss + target_domain_loss
        domain_loss.backward()
        optimizers['discriminator'].step()
        
        print(target_domain_pred.detach().cpu().numpy()[0])
        print(source_domain_pred.detach().cpu().numpy()[0])
        total_hit += torch.sum(torch.round(source_domain_pred) == S_labels).item()
        total_hit += torch.sum(torch.round(target_domain_pred) == T_labels).item()
        total_num += source_data.shape[0]
        total_num += target_data.shape[0]
        #torch.cuda.empty_cache()
        #train extractor
        T_labels = torch.ones([target_data.shape[0],1]).to(device)
        target_domain_pred = models['discriminator'](models['lstm'](models['conv'](target_data)))
        target_domain_loss = domain_criterion(target_domain_pred,T_labels)
        
        source_logits = (models['fc']\
                              (models['lstm']\
                                (models['conv'](source_data))))

        target_logits = models['lstm']\
                                (models['conv'](target_data))
        predict_target = models['regression'](models['fc'](target_logits)).squeeze()
        target_loss = criterion(predict_target,target_label)
        mmd_loss = 0
        if lamb3 != 0:
          mmd_loss = mmd(torch.reshape(target_logits,(target_logits.shape[0],-1)),torch.reshape(source_logits,(source_logits.shape[0],-1)))
        loss = lamb1*target_loss + lamb2 * (target_domain_loss)  + lamb3 * mmd_loss
        loss.backward()
        optimizers['conv'].step()
        optimizers['lstm'].step()
        source_sample += len(source_data)
        target_sample += len(target_data)
        loss_train += target_loss.item()
        loss_domain += domain_loss.item()
        loss_target_domain += target_domain_loss.item()
        #if ((epoch+1) % eval_interval) == 0:
          #plot_result(source_label, predict_label, save_image='train', 
                  #test_name=source_data_path[7:-1] + temp[temp_idx] + '_epoch_' + str(epoch))
      loss_train = loss_train/(source_sample)
      loss_domain = loss_domain/(target_sample)
      acc = (float)(total_hit)/total_num
  
      print('epoch {}:loss {} target_domain_loss {}domain_loss {} domain_acc {} {}/{}'.format(epoch, loss_train,loss_target_domain,loss_domain,acc,total_hit,total_num))
      if (loss_train < loss_min) & (ifsave==True):
        loss_min = loss_train
        path = rundir + '/saved_model/best.pt'
        #save_model(models, optimizers, loss_min, seed,path)
        #print('min loss:{} saved model'.format(loss_min))
      #####
      #plt tsne
      #####
      if ((epoch+1) % eval_interval) == 0:
            lossfile = rundir + '/errors/domain_loss.txt'
            with open(lossfile,'a') as f:
              f.write('epoch ' + str(epoch) + ' ' + str(loss_target_domain) + '\n')
            tqdm_mix = tqdm(zip(source_loader, target_loader),desc='epoch '+str(epoch))
            data,feat,labels =[], [],[]
            for i, ((source_data, source_label), (target_data, target_label)) in enumerate(tqdm_mix):
              source_data = source_data.to(device)
              target_data = target_data.to(device)
              source_len = source_data.shape[0]
              target_len = target_data.shape[0]
              feat_S,feat_T = models['conv'](source_data).cpu().detach().numpy(),models['conv'](target_data).cpu().detach().numpy()
              data.append(source_data.reshape(source_len,-1).cpu().detach().numpy())
              data.append(target_data.reshape(target_len,-1).cpu().detach().numpy())
              feat.append(feat_S.reshape(source_len,-1))
              feat.append(feat_T.reshape(target_len,-1))
              S_labels = np.zeros([source_len])
              T_labels = np.zeros([target_len])
              T_labels = T_labels + 5
              labels.append(S_labels.astype(int))
              labels.append(T_labels.astype(int))
            print(len(feat))
            feat = np.concatenate(feat,axis=0)
            data = np.concatenate(data,axis=0)
            labels = np.concatenate(labels)
            feat_path = rundir + '/tsne/feat_epoch' + str(epoch) + '.jpg'
            plot_tsne(feat,labels,feat_path)
            data_path = rundir + '/tsne/data_epoch' + str(epoch) + '.jpg'
            plot_tsne(data,labels,data_path)
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
          print('target train')
      for i, data in enumerate(target_test_loader_2):
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
        
        if ((epoch+1) % eval_interval) == 0:
          test_name=target_data_path[-4:-1] + target_temp + target_train_set[i]
          plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
          error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
          print(error)
          save_error(rundir,error,test_name,'test')
          #plot_result(y_test, y_predict, epoch, i, set_name=temps[temp_idx], save_image='test')
      
      
      #tqdm_test = tqdm(target_test_loader, desc='target data test')
      loss_mae = 0
      loss_rmse = 0
      loss_max = 0
      if ((epoch+1) % eval_interval) == 0:
          print('source test')
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
        
        if ((epoch+1) % eval_interval) == 0:
          test_name=source_data_path[-4:-1] + source_temp + source_test_set[i]
          plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
          error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
          print(error)
          save_error(rundir,error,test_name,'test')
          
      loss_mae = 0
      loss_rmse = 0
      loss_max = 0
      if ((epoch+1) % eval_interval) == 0:
          print('target test')
      for i, data in enumerate(target_test_loader):
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
        if epoch > 1: 
          loss_avg = (loss_mae+loss_rmse)/2
          min_avg = (min_mae[i] + min_rmse[i])/2
          if loss_avg < min_avg:
            min_max[i] = loss_max
            min_rmse[i] = loss_rmse
            min_mae[i] = loss_mae
            test_name=target_data_path[-4:-1] + target_temp + target_test_set[i]+'best'
            plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
            error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
            print(error)
            save_error(rundir,error,test_name,'test')
            save_min(rundir,min_mae,min_rmse,min_max,epoch,domain_loss,domain_acc)
            loss_min = loss_train
            path = rundir + '/saved_model/best.pt'
            save_model(models, optimizers, loss_min, seed,path)
            print('min avg loss:{} saved model'.format(loss_avg))
            
        if ((epoch+1) % eval_interval) == 0:
          test_name=target_data_path[-4:-1] + target_temp + target_test_set[i]
          plot_result(rundir,y_test, y_predict, save_image='test',test_name=test_name)
          error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
          print(error)
          save_error(rundir,error,test_name,'test')

      #####
      loss_iter_domain.append(loss_domain)
      loss_iter_predictor.append(loss_train)
      loss_iter_test.append(loss_test)
      loss_iter_mae.append(loss_mae)
      loss_iter_rmse.append(loss_rmse)
      loss_iter_max.append(loss_max)
      loss_iter_domain_acc.append(acc)
      #if ( ((epoch+1) % eval_interval)==0 ) & (ifsave==True):
      #  save_model(models, optimizers, loss_min, seed, model_path='./saved_model/epoch'+str(epoch)+'.pt')
    plot_train_loss(rundir,loss_iter_domain, loss_iter_predictor, loss_iter_domain_acc, epochs)
    plot_test_loss(rundir,loss_iter_mae, loss_iter_rmse, loss_iter_max, epochs)
    print(min_mae,min_rmse,min_max)