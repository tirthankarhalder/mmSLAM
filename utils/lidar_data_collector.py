import threading
import os 
import time 
import struct 
import csv
import numpy as np
from datetime import datetime
import PyLidar3
import math
 

header = list(np.arange(0,360,1))
header = ['datetime',*header]
x=[]
y=[]
for _ in range(360):
    x.append(0)
    y.append(0)
def collect_lidar_data(duration, filename,LidarPort):
    port = LidarPort #linux
    Obj = PyLidar3.YdLidarX4(port) #PyLidar3.your_version_of_lidar(port,chunk_size) 
    # Construct the full path with the desired directory
    directory_path = os.path.join('./datasets/', 'lidar_data')
    full_path = os.path.join(directory_path, filename)
    print(full_path)    
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("lidar_data directory is created")
    
    # Check if the file exists, delete it if it does
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"File {full_path} already existed. Overwriting...")


    with open(full_path, "w") as f:
        csv.DictWriter(f, fieldnames=header).writeheader()
    if(Obj.Connect()):
        # print(Obj.GetDeviceInfo())
        gen = Obj.StartScanning()
        t = time.time() # start time 
        # path = file_create(filepath)
        end_time = time.time() + duration
        
        while(time.time() < end_time):
            print("Time Left: ",duration-time.time())
            data = next(gen)
            for angle in range(0,360):
                if(data[angle]>1000):
                    x[angle] = data[angle] * math.cos(math.radians(angle))
                    y[angle] = data[angle] * math.sin(math.radians(angle))
            dict_dumper = {'datetime': datetime.now()}
            dict_dumper.update(data)
            #print(dict_dumper)
            with open(full_path, "a") as f:
                writer = csv.DictWriter(f, header)
                writer.writerow(dict_dumper)
            time.sleep(0.5)
        Obj.StopScanning()
        Obj.Disconnect()
    else:
        print("Error connecting to device")

