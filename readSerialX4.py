import PyLidar3
import time # Time module
import csv
import numpy as np
from datetime import datetime 
from pathlib import Path
import math
import matplotlib.pyplot as plt
#Serial port to which lidar connected, Get it from device manager windows
#In linux type in terminal -- ls /dev/tty* 
# port = input("Enter port name which lidar is connected:") #windows
port = "/dev/ttyUSB0" #linux
Obj = PyLidar3.YdLidarX4(port) #PyLidar3.your_version_of_lidar(port,chunk_size) 
header = list(np.arange(0,360,1))
header = ['datetime',*header]


filepath = "./dataX4/"
filepath += time.strftime("%Y%m%d_%H%M%S")
Path(filepath).mkdir(parents=True,exist_ok=True)
Path(filepath + '/fig').mkdir(parents=True,exist_ok=True)


x=[]
y=[]
for _ in range(360):
    x.append(0)
    y.append(0)

def file_create(path):
    filename = path +  "/data.csv"  
    with open(filename, "w") as f:
        csv.DictWriter(f, fieldnames=header).writeheader()
    return filename


if __name__ == "__main__":
    if(Obj.Connect()):
        # print(Obj.GetDeviceInfo())
        gen = Obj.StartScanning()
        t = time.time() # start time 
        path = file_create(filepath)
        while True:
            data = next(gen)
            for angle in range(0,360):
                if(data[angle]>1000):
                    x[angle] = data[angle] * math.cos(math.radians(angle))
                    y[angle] = data[angle] * math.sin(math.radians(angle))


            fig = plt.figure(figsize=(12,7))  
            plt.ylim(-9000,9000)
            plt.xlim(-9000,9000)
            plt.scatter(x,y,c='r',s=2)  
            plt.savefig(filepath + "/fig/"+time.strftime("%Y%m%d_%H%M%S")+".png")
            # plt.show()
            plt.close()
            dict_dumper = {'datetime': datetime.now()}
            dict_dumper.update(data)
            print(dict_dumper)
            with open(path, "a") as f:
                writer = csv.DictWriter(f, header)
                writer.writerow(dict_dumper)
            time.sleep(0.5)
        Obj.StopScanning()
        Obj.Disconnect()
    else:
        print("Error connecting to device")

