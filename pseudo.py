from locale import normalize
from time import time
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn import datasets
from sklearn.manifold import TSNE
from utils import *
from mydata import *
from models import *
import numpy as np
import math
import scipy.io as sio
import argparse

#use pre-trained source model to generate pseudo labels for unlabelled target data
def main(temp,model,file):
    print('temp:',temp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    models = {}
    models['conv'] = conv()
    models['lstm'] = lstm()
    models['fc'] = fc()
    models['regression'] = regression()
    for model in models:
        models[model].to(device)
        models[model].eval()
    load_model_path = model
    ckpt = torch.load(load_model_path, map_location=device)
    model = ['conv','lstm','fc','regression']
    for m in model:
        models[m].load_state_dict(ckpt[m])
    seed = ckpt['seed']
    print('load model')
    print('load seed={}'.format(seed))
    init_seed(seed)

    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    
    data_path = Pan_data_path
    
    test_set = []
    test_set.append(file)
    print(test_set)
    test_data = Mydataset(data_path, temp, test_set,mode='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    print('load data')
    for i, data in enumerate(test_loader):
        print('data ',i)
        x_test, y_test = data
        x_test, y_test = x_test.squeeze(), y_test.squeeze()
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        with torch.no_grad():
            y_predict = models['regression']\
                                (models['fc']\
                                (models['lstm']\
                                    (models['conv'](x_test)))).squeeze()
            print(y_predict.shape)
    y_predict = y_predict.flatten().cpu().numpy()
    y_test = y_test.flatten().cpu().numpy()

    
    y_diff = []
    
    x = 200
    for i in range(len(y_predict)):
        if (i+x) > len(y_predict)-1:
            break
        a = ( (y_predict[i] - y_predict[i+x]) + \
            (y_predict[i] - y_predict[i+int(x/3)]) + \
            (y_predict[i] - y_predict[i+int(2*x/3)])
            ) / 2
        y_diff.append(abs(a))
    y_diff = np.asarray(y_diff)
    idx_set = set(list(range(len(y_diff))))
    
    thresh = 0.08

    for i,e in enumerate(y_diff):
        if e > thresh:
            b1 = int(x/10)
            b2 = int(x)
            front_idx = max(0,i-b1)
            back_idx = min(len(y_diff),i+b2)
            for j in range(front_idx,back_idx):
                idx_set.discard(j)

    y_predict_cut = y_predict[list(idx_set)]
    y_test_cut = y_test[list(idx_set)]
    y_diff_cut= y_diff[list(idx_set)]
    y_diff2 = -(np.diff(y_predict_cut))
    pairs = []
    sets_list = []
    idx_list = list(idx_set)
    split_point = []
    split_point.append(0)
    for i,e in enumerate(y_diff2):
        if abs(y_diff2[i]) > 0.08:
            split_point.append(i)
            print(i,idx_list[i],y_diff2[i])
    for i,e in enumerate(split_point):
        
        if i == (len(split_point)-1):
            beg,end = min(split_point[i]+2,len(idx_list)-2),len(idx_list)-2
        else:
            beg,end = split_point[i]+2,split_point[i+1]-2
            if y_diff2[split_point[i+1]] < 0:
                continue
        
        pairs.append((beg,end))
        split_set = set()
        print(idx_list[beg],idx_list[end])
        for k in range(idx_list[beg],idx_list[end]):
            if k in idx_set:
                split_set.add(k)
        sets_list.append(split_set)
    
    fig_cut = 'fig_' + temp + file + '_pseudo*_cut.jpg'
    fig = 'fig_' + temp + file + '_pseudo*.jpg'
    for i,split_set in enumerate(sets_list):
        fig_split = 'fig_' + temp + file + '_pseudo*_split_' + str(i+1) + '.jpg'
        plt.figure()
        y_predict_split = y_predict[sorted(split_set)]
        y_test_split = y_test[sorted(split_set)]
        split_diff = abs(y_predict_split - y_test_split)
        plt.plot(y_predict_split,label='predict',color='red')
        plt.plot(y_test_split,label='label',color='black')
        #plt.plot(y_diff,label='diff',color='blue')
        plt.plot(split_diff,label='diff',color='green')
        plt.legend()
        plt.savefig(fig_split)
        plt.close()
        path = './your fold' + temp + '/' + temp + 'degC_' + file + '_PF.mat'
        mat = sio.loadmat(path)
        time,current,voltage,battery_temp,ah = mat['time'][sorted(split_set)],mat['current'][sorted(split_set)],mat['voltage'][sorted(split_set)],mat['temp'][sorted(split_set)],mat['ah']
        file = './your fold' + temp + '/' + temp + 'degC_' + file + '_pseudo*_' + str(i+1) + '_PF.mat'
        ah = y_predict_split
        ah = ah.reshape(-1,1)
        sio.savemat(file,{'time':time,'current':current,'voltage':voltage,'temp':battery_temp,'ah':ah})
        print(time.shape,y_predict_split.shape)
    #draw figures
    

    print('<{}:{}'.format(str(thresh),np.sum(y_diff<thresh)))
    plt.figure()
    plt.plot(y_predict_cut,label='predict',color='red')
    plt.plot(y_test_cut,label='label',color='black')
    #plt.plot(y_diff,label='diff',color='blue')
    plt.plot(y_diff2,label='diff2',color='green')
    plt.legend()
    plt.savefig(fig_cut)
    plt.close()

    plt.figure()
    plt.plot(y_predict,label='predict',color='red')
    plt.plot(y_test,label='label',color='black')
    plt.plot(y_diff,label='diff',color='blue')
    plt.legend()
    plt.savefig(fig)
    plt.close()

    print(y_predict.shape,y_test.shape)
    
    #calculate error
    loss_mse = np.sum((y_predict-y_test)**2) / len(y_test)
    loss_mae = np.sum( abs(y_predict-y_test) ) / len(y_test)
    loss_rmse = math.sqrt(loss_mse)
    loss_max = max(abs(y_predict-y_test))
    error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
    print(data_path,test_set[0])
    print(error)

    loss_mse = np.sum((y_predict_cut - y_test_cut)**2) / len(idx_set)
    loss_mae = np.sum( abs(y_predict_cut - y_test_cut) ) / len(idx_set)
    loss_rmse = math.sqrt(loss_mse)
    loss_max = max(abs(y_predict_cut - y_test_cut))
    error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
    print(data_path,test_set[0])
    print(error)
    
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate pseudo label')
    parser.add_argument('--temp',type=str,default='n20')
    parser.add_argument('--model',type=str,default='models/pre-n10.pt')
    parser.add_argument('--file',type=str,default='your file')
    args = parser.parse_args()
    main(args.temp,args.model,args.file)