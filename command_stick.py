import serial
import time
import subprocess
import sys
import os
import csv
from datetime import datetime
import argparse
import cv2
import pyrealsense2 as rs
import numpy as np
# import board 
# import adafruit_mpu6050
import threading
from utils.imu_data_collector import collect_data
from utils.video_cap import capture_video
#from git import Repo
#from utils import push

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

def execute_depth_camera():
    header = [
        "datetime",
        "frame_number",
        "x",
        "y",
        "z"
        
    ]
    filename = "depth.csv"  
    with open(filename, "w") as f:
        csv.DictWriter(f, fieldnames=header).writeheader()
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Create point cloud object
            pc = rs.pointcloud()
            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)
            # print(points.get_vertices())

            dict_dumper = {'datetime': datetime.now()}
            data = {
                "datetime" : datetime.now(),
                "frame_number": points.get_frame_number(),
                "x":np.asanyarray(points.get_vertices())['f0'],
                "y":np.asanyarray(points.get_vertices())['f1'],
                "z":np.asanyarray(points.get_vertices())['f2']
            }
            
            dict_dumper.update(data)
            path = "./depth.csv"
            with open(path, "a") as f:
                writer = csv.DictWriter(f, header)
                writer.writerow(dict_dumper)
            # print("Frame #:",points.get_frame_number())
            # print("Frame size:", points.get_data_size())
            # print("Frame profile:", points.profile)
            # # depth = np.asanyarray(points.get_data())
            # depth = np.asanyarray(points.as_points().get_data()) 
    
            # print("Depth shape", depth.shape)
            # print("Depth", depth)  


            # Save point cloud data as a .ply file
            # points.export_to_ply("point_cloud.ply", color_frame)
            # print("Saved point cloud to 'point_cloud.ply'")

            # Display the color image
            # cv2.imshow('RealSense', color_image)

            # # Exit on 'q' key press
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        # Stop streaming
        pipeline.stop()

        # Close OpenCV windows
        # cv2.destroyAllWindows()

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
    parser.add_argument('-imu', action='store_true')
    parser.add_argument('-depth', action='store_true')
    ans1=input("Have you connected the ethernet to Jetson? yes/no: ")
    camera_pass = False
    args = parser.parse_args()
    mac_command = f"sudo macchanger --mac=e8:b5:d0:fe:9c:67 eth0"
    print(mac_command)
    os.system(mac_command)
    if(args.camera):
        ans3=input("Have you connected camera cable? yes/no: ")
        if(ans3=="yes"):
            camera_pass = True
    elif(not args.camera):
        camera_pass= True
    if ans1=='yes' and camera_pass: 
        c_program_path = "/home/stick/mmSLAM/data_collect_mmwave_only" 
        image_folder_path = "./scene_annotation/"
        now = datetime.now()
        date_string = str(now.strftime('%Y-%m-%d_%H_%M_%S'))
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
        file_name="drone_"+date_string+"_"+".bin"
        image_name = "drone_"+date_string+"_"+".jpg"
        c_program_args=[file_name,n_frames]
        if(args.camera):
            capture_frame_and_save(image_folder_path, image_name)
        # video_filename =  date_string+"_"+pwm_value+".mp4"
        # video_thread = threading.Thread(target=capture_video, args=(imu_duration, video_filename))
        # video_thread.start()
        if(args.imu):
            imu_duration = (int(n_frames)+5)*int(periodicity) / 1000; #periodicity is in ms (collect for 5 extra frames)
            imu_filename = "drone_"+date_string+"_imu.bin"
            imu_thread = threading.Thread(target=collect_data, args=(imu_duration, imu_filename))
            imu_thread.start()     
        execute_c_program(c_program_path,c_program_args)
        if(args.imu):
            imu_thread.join()    
            # video_thread.join()
        if(args.depth):
            execute_depth_camera()
        ans_to_keep=input('Do you want to keep the reading? yes/no : ')
        if(ans_to_keep=='no'):
            os.system(f"rm {file_name}")
            print(f"{file_name} deleted successfully")
            os.system(f"rm ./imu_data/{imu_filename}")
            print(f"./imu_data/{imu_filename} deleted successfully")
            sys.exit()
        #os.system(f"mv {file_name} /media/stick/Seagate\ Backup\ Plus\ Drive/")
        #if (args.imu):
            #os.system(f"mv ./imu_data/{imu_filename} /media/stick/Seagate\ Backup\ Plus\ Drive/imu_data/")
        file_path="dataset_stick.csv"
        data=[file_name,n_frames,n_chirps,tc,adc_samples,sampling_rate,periodicity,l,r0,descri]
        if r0==l:
            data.append('Straight')
        else:
            data.append('Oblique')
        
        with open(file_path,'a',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(data)
            print('Data appended successfully')

