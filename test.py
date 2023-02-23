from torch.utils.data import Dataset, DataLoader
from mydata import *
from utils import *
from models import *
from tqdm import tqdm
import math
def test(rundir,temp,models, data_path, test_set, seed=0, device_type=('cuda' if torch.cuda.is_available() else 'cpu'),load_model_path='./saved_model/best.pt'):
  device = torch.device(device_type)
  rundir = mkdir(rundir)
  for model in models:
    models[model].to(device)
    models[model].eval()
  ckpt = torch.load(load_model_path, map_location=device)
  model = ['conv','lstm','fc','regression']
  for m in model:
    models[m].load_state_dict(ckpt[m])
  seed = ckpt['seed']
  print('load model')
  print('load seed={}'.format(seed))
  init_seed(seed)

  #criterion for test
  criterion_mae = nn.L1Loss()
  criterion_mse = nn.MSELoss()

  test_data = Mydataset(data_path, temp, test_set,mode='test')
  test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

  ##########
  #test
  ##########
  #tqdm_test = tqdm(test_loader, desc='temp %s'%temp)
      
  loss_mae = 0
  loss_rmse = 0
  loss_max = 0
  for i, data in enumerate(test_loader):
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
      
    error = 'mae = ' + str(loss_mae) + ' rmse = ' + str(loss_rmse) +  ' max = ' + str(loss_max)
    test_name=data_path[-4:-1]+temp+test_set[i]
    print(data_path,test_set[i])
    print(error)
    plot_result(rundir,y_test, y_predict, save_image='test', test_name=test_name)
    save_error(rundir,error,test_name,'test')

    