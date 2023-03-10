import os
from sys import prefix
import scipy.io as sio
temps=['25','10']#just for example

#read from data file of Panasonic 18650PF dataset, trans the file from 
#you need to rewrite the code if you have different dataset.
def read_and_save(path):  
    time = []
    voltage = []
    current = []
    ah = []
    temp = []
    for f in os.listdir(path):
        file = path +'/' + f
        print(file)
        data = sio.loadmat(file)
        time = (data['meas']['Time'][0][0])
        current = (data['meas']['Current'][0][0])
        voltage = (data['meas']['Voltage'][0][0])
        temp = (data['meas']['Battery_Temp_degC'][0][0])
        ah = (data['meas']['Ah'][0][0])
        sio.savemat(file,{'time':time,'current':current,'voltage':voltage,'temp':temp,'ah':ah})
       
#re-sample the data 
def sample(path):
        interval = 10
        for f in os.listdir(path):
            if f.endswith('_PF.mat'):
                print(f)
                file = path + f
                data = sio.loadmat(file)
                print(len(data['time']))
                time = data['time'][::interval]
                print(len(time))
                current = data['current'][::interval]
                voltage = data['voltage'][::interval]
                temp = data['temp'][::interval]
                ah = data['ah'][::interval]
                sio.savemat(file,{'time':time,'current':current,'voltage':voltage,'temp':temp,'ah':ah})

#read from the file and record the range for data normalization later
def read_mat(path):
        min_current, min_voltage, min_ah, min_temp = 100,100,100,100
        max_current,max_voltage,max_ah, max_temp = -100,-100,-100,-100
        with open(path+'each_file_data.txt','w') as out:
            out.write('')
        for f in os.listdir(path):
            if f.endswith('.mat') & (not f.endswith('LG.mat')):
                print(f)
                file = path + f
                data = sio.loadmat(file)
                time = data['time']
                print(len(time))
                print(time[:3])
                idx = 0
                for i in range(len(time)):
                    if (time[i+1]-time[i] < 2):
                        idx = i
                        print(idx)
                        print(time[i],time[i+1])
                        break
                time = time[idx:]
                current = data['current'][idx:]
                voltage = data['voltage'][idx:]
                temp = data['temp'][idx:]
                ah = data['ah'][idx:]
                if min_current > min(current):
                    min_current = min(current)
                if min_voltage > min(voltage):
                    min_voltage = min(voltage)
                if min_ah > min(ah):
                    min_ah = min(ah)
                if min_temp > min(temp):
                    min_temp = min(temp)
                if max_current < max(current):
                    max_current = max(current)
                if max_voltage < max(voltage):
                    max_voltage = max(voltage)
                if max_ah < max(ah):
                    max_ah = max(ah)
                if max_temp < max(temp):
                    max_temp = max(temp) 
                r1 = 'current'+str(min(current))+str(max(current))+'\n'
                r2 = 'voltage'+str(min(voltage))+str(max(voltage))+'\n'
                r3 = 'temp'+str(min(temp))+str(max(temp))+'\n'
                r4 = 'ah'+str(min(ah))+str(max(ah))+'\n'
                with open(path+'each_file_data.txt','a') as out:
                    out.write(f+'\n')
                    out.write(r1)
                    out.write(r2)
                    out.write(r3)
                    out.write(r4)
                    out.write('\n')
            r1 = 'current'+str(min_current)+str(max_current)+'\n'
            r2 = 'voltage'+str(min_voltage)+str(max_voltage)+'\n'
            r3 = 'temp'+str(min_temp)+str(max_temp)+'\n'
            r4 = 'ah'+str(min_ah)+str(max_ah)+'\n'
            with open(path+'data.txt','w') as out:
                out.write(r1)
                out.write(r2)
                out.write(r3)
                out.write(r4)
        print(r1,r2,r3,r4)
             
"""data normalization into [0,1]
for every profile, the range is different. 
You need to record range for every file and revise the range
"""
def transform(path):
    range_current = [-17.6,6.003]
    range_voltage = [2.799,4.209]
    range_temp = [23.555,26.82]
    range_ah = [-2.591, 0.00131]
    #above range need to be revised according to different file
    for f in os.listdir(path):
        file = path + f
        mat = sio.loadmat(file)
        s = f.split('_')
        newfile = path + s[0] + '_' + s[1] + '.mat'
        time,current,voltage,battery_temp,ah = mat['time'],mat['current'],mat['voltage'],mat['temp'],mat['ah']
        current = (current-range_current[0])/(range_current[1]-range_current[0])
        voltage = (voltage-range_voltage[0])/(range_voltage[1]-range_voltage[0])
        battery_temp = (battery_temp-range_temp[0])/(range_temp[1]-range_temp[0])
        ah = (ah - range_ah[0])/(range_ah[1]-range_ah[0])
        sio.savemat(newfile,{'time':time,'current':current,'voltage':voltage,'temp':battery_temp,'ah':ah})

#split data into train set and test set
def cutset(data_path): 
  for t in temps:
    path = data_path + t + '/'
    
    for f in os.listdir(path):
      if f.endswith('.mat'):
        file = path + f
        s = f.split('.mat')[0]
        trainfile = path + s + '_train.mat'
        testfile =  path + s + '_test.mat'
        print(file)
        data = sio.loadmat(file)
        time = data['time']
        current = data['current']
        voltage = data['voltage']
        temp = data['temp']
        ah = data['ah']
        
        idx = int(len(ah)*0.3)
        print(len(ah))
        print(idx)
        sio.savemat(trainfile,{'time':time[:idx],'current':current[:idx],'voltage':voltage[:idx],'temp':temp[:idx],'ah':ah[:idx]})
        sio.savemat(testfile,{'time':time[idx:],'current':current[idx:],'voltage':voltage[idx:],'temp':temp[idx:],'ah':ah[idx:]})


if __name__ == '__main__':
    path = 'your data path'
    read_and_save(path)
    sample(path)
    read_mat(path)
    transform(path)
    cutset(path)

