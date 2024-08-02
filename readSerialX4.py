import PyLidar3
import time # Time module
import csv
import numpy as np
from datetime import datetime 
#Serial port to which lidar connected, Get it from device manager windows
#In linux type in terminal -- ls /dev/tty* 
# port = input("Enter port name which lidar is connected:") #windows
port = "/dev/ttyUSB0" #linux
Obj = PyLidar3.YdLidarX4(port) #PyLidar3.your_version_of_lidar(port,chunk_size) 
header = list(np.arange(0,360,1))
header = ['datetime',*header]

def file_create():
    # filename = "sample"
    # filename = os.path.abspath("") 
    filename = "./dataX4/"
    filename += time.strftime("%Y%m%d_%H%M%S")
    filename += ".csv"    
    with open(filename, "w") as f:
        csv.DictWriter(f, fieldnames=header).writeheader()
    return filename


if __name__ == "__main__":
    if(Obj.Connect()):
        print(Obj.GetDeviceInfo())
        gen = Obj.StartScanning()
        t = time.time() # start time 
        filepath = file_create()
        # with open(file_create(),mode='w',newline='') as file:
        # while (time.time() - t) < 5: #scan for 30 seconds
                # print(next(gen))
        while True:
            data = next(gen)
            dict_dumper = {'datetime': datetime.now()}
            dict_dumper.update(data)
            print(dict_dumper)
            with open(filepath, "a") as f:
                writer = csv.DictWriter(f, header)
                writer.writerow(dict_dumper)
            time.sleep(0.5)
        Obj.StopScanning()
        Obj.Disconnect()
    else:
        print("Error connecting to device")

