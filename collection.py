import argparse
import datetime
import os
import csv
from  pathlib import Path
import numpy as np
import threading
import sys
import subprocess
import cv2
import asyncio
from mavsdk import System
import PyLidar3
import matplotlib.pyplot as plt
import time
import math



#define the header for different modalities
lidarHeader = list(np.arange(0,360,1))
lidarHeader = ['datetime',*lidarHeader]


# defined global path
today = datetime.date.today()
date_string = today.strftime('%Y-%m-%d-%H-%M-%S')
filepath = "./datasets/"
filepath+=date_string
Path(filepath).mkdir(parents=True,exist_ok=True)


lidarFile = filepath +  "/lidardata.csv"  
with open(lidarFile, "w") as f:
    csv.DictWriter(f, fieldnames=lidarHeader).writeheader()



#defin global variable
x=[]
y=[]
for _ in range(360):
    x.append(0)
    y.append(0)
generate_images = False
port = "/dev/ttyUSB0" #linux
Obj = PyLidar3.YdLidarX4(port) 


def execute_c_program(c_program_path, c_program_args):
    command=[c_program_path] + c_program_args
    print("command: ", command)
    # Execute the C program
    try:
        print("Executing C program...")
        result = subprocess.run(command, check=True)
        print("C program executed successfully.")
    except subprocess.CalledProcessError as e:
        #print(f"Error executing C program: {e}")
        pass


def capture_frame_and_save(folder_path, image_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera, exiting thread")
        sys.exit()
        return
    ret, frame = cap.read()
    cap.release()
    if ret:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_path = os.path.join(folder_path, image_name)
        cv2.imwrite(image_path, frame)
        print("Image saved successfully:", image_path)
    else:
        print("Error: Failed to capture frame")


async def print_status_text():
    drone = System()
    await drone.connect(system_address="serial:///dev/serial0:921600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    # async for ned in drone.telemetry.positionbody():
    #     print("NED:", ned)


    async for odom in drone.telemetry.odometry():
        print("Odometry:", odom)

def file_create():
    # filepath += time.strftime("%Y%m%d_%H%M%S")
    # Path(filepath + '/fig').mkdir(parents=True,exist_ok=True)
    
    # Dataset creation for lidar
    lidarFile = filepath +  "/lidardata.csv"  
    with open(lidarFile, "w") as f:
        csv.DictWriter(f, fieldnames=lidarHeader).writeheader()

    # Dataset creation for Radar



    return lidarFile



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for params')
    parser.add_argument('-nf', '--nframes', type=int, help='Number of frames')
    parser.add_argument('-nc', '--nchirps', type=int, help='Number of chirps in a frame, usually 182')
    parser.add_argument('-tc', '--timechirp', type=int, help='Chrip time is microseconds, usually 72')
    parser.add_argument('-s', '--samples', type=int, help='Number of ADC samples, or range bins, usually 256')
    parser.add_argument('-r', '--rate', type=int, help='Sampling rate, usually 4400')
    parser.add_argument('-tf', '--timeframe', type=int, help='Periodicity or Frame time in milliseconds')
    parser.add_argument('-l', '--length', type=int, help='Initial length')
    parser.add_argument('-r0', '--radial', type=int, help='Initial radial distance')
    parser.add_argument('-d', '--descp', type=str, help='Data description')
    parser.add_argument('-camera', action='store_true')
    parser.add_argument('-lidar', action='store_true')
    parser.add_argument('-imu', action='store_true')
    parser.add_argument('-depth', action='store_true')
    ans1=input("Have you connected the ethernet to Jetson? yes/no: ")
    camera_pass = False
    args = parser.parse_args()
    mac_command = f"sudo macchanger --mac=08:97:98:70:B9:13 eth0"
    print(mac_command)
    os.system(mac_command)
    if(args.camera):
        ans3=input("Have you connected camera cable? yes/no: ")
        if(ans3=="yes"):
            camera_pass = True
    elif(not args.camera):
        camera_pass= True

    if ans1=='yes' and camera_pass: 
        c_program_path = "/home/stick/mmPhase/data_collect_mmwave_only" 
        image_folder_path = filepath + "/scene_annotation/"

        # now = datetime.now()
        # date_string = str(now.strftime('%Y-%m-%d_%H_%M_%S'))
        n_frames = str(args.nframes)
        n_chirps = str(args.nchirps)
        tc       = str(args.timechirp)
        adc_samples = str(args.samples)
        sampling_rate = str(args.rate)
        periodicity = str(args.timeframe)
        l = str(args.length)
        r0 = str(args.radial)
        descri = args.descp
        date_string+="_" + descri
        radarFile_name=filepath + "/spectral_"+descri+".bin"
        image_name = filepath + "/spectral_"+descri+".jpg"
        c_program_args=[radarFile_name,n_frames]
        # video_filename =  date_string+"_"+pwm_value+".mp4"
        # video_thread = threading.Thread(target=capture_video, args=(imu_duration, video_filename))
        # video_thread.start()
        if(args.camera):
            capture_frame_and_save(image_folder_path, image_name) 

        if args.lidar:
            if(Obj.Connect()):
                print(Obj.GetDeviceInfo())
                gen = Obj.StartScanning()
                t = time.time() # start time 
                # path = file_create(filepath)
                while True:
                    data = next(gen)
                    for angle in range(0,360):
                        if(data[angle]>1000):
                            x[angle] = data[angle] * math.cos(math.radians(angle))
                            y[angle] = data[angle] * math.sin(math.radians(angle))

                    if generate_images:
                        fig = plt.figure(figsize=(12,7))  
                        plt.ylim(-9000,9000)
                        plt.xlim(-9000,9000)
                        plt.scatter(x,y,c='r',s=2)  
                        plt.savefig(filepath + "/fig/"+time.strftime("%Y%m%d_%H%M%S")+".png")
                        # plt.show()
                        plt.close()
                    dict_dumper = {'datetime': datetime.now()}
                    dict_dumper.update(data)
                    # print(dict_dumper)
                    
                    with open(lidarFile, "a") as f:
                        writer = csv.DictWriter(f, lidarHeader)
                        writer.writerow(dict_dumper)
                    time.sleep(0.5)
                Obj.StopScanning()
                Obj.Disconnect()
            else:
                print("Error connecting to device")


        if(args.imu):
            asyncio.run(print_status_text())
            # imu_duration = (int(n_frames)+5)*int(periodicity) / 1000; #periodicity is in ms (collect for 5 extra frames)
            # imu_filename = "stick_"+date_string+"_imu.bin"
            # imu_thread = threading.Thread(target=collect_data, args=(imu_duration, imu_filename))
            # imu_thread.start() 
        
        
        execute_c_program(c_program_path,c_program_args)
         
        if(args.depth):
            pass



        ans_to_keep=input('Do you want to keep the reading? yes/no : ')     
        if(ans_to_keep=='no'):
            # os.system(f"rm {filepath}")
            os.system(f"rd /s {filepath}")
            print(f"{filepath} deleted successfully")
            # os.system(f"rm ./imu_data/{imu_filename}")
            # print(f"./imu_data/{imu_filename} deleted successfully")
            sys.exit()   


        file_path=filepath + "/dataset_spectral.csv"
        data=[radarFile_name,n_frames,n_chirps,tc,adc_samples,sampling_rate,periodicity,l,r0,descri]
        if r0==l:
            data.append('Straight')
        else:
            data.append('Oblique')
        
        with open(file_path,'a',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(data)
            print('Data appended successfully')




