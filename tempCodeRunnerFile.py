import PyLidar3
import time # Time module
import csv
#Serial port to which lidar connected, Get it from device manager windows
#In linux type in terminal -- ls /dev/tty* 
# port = input("Enter port name which lidar is connected:") #windows
port = "/dev/ttyUSB1" #linux
Obj = PyLidar3.YdLidarX4(port) #PyLidar3.your_version_of_lidar(port,chunk_size) 
def file_create():
    # filename = "sample"
    # filename = os.path.abspath("") 
    filename = "./dataX4/"
    filename += time.strftime("%Y%m%d_%H%M%S")
    filename += ".csv"
    return filename


if __name__ == "__main__":
    if(Obj.Connect()):
        print(Obj.GetDeviceInfo())
        gen = Obj.StartScanning()
        t = time.time() # start time 
        with open(file_create(),mode='w',newline='') as file:
            while (time.time() - t) < 5: #scan for 30 seconds
                # print(next(gen))
                data =  dict(next(gen))
                # writer = csv.writer(file)
                writer = csv.DictWriter(file,data.keys())
                # print(list(data))
                writer.writerows(data)
                print(data)
                time.sleep(0.5)
        Obj.StopScanning()
        Obj.Disconnect()
    else:
        print("Error connecting to device")

