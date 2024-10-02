import threading
import os 
import time 
import struct 
import csv
import cv2
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
import pickle
import h5py
import queue

buffer_size = 5
buffer = queue.Queue(maxsize=buffer_size)
lock = threading.Lock()
'''
    pklObjects->stack of all buffer
    buffer->stack of all frames
    frames->stack of poincloud,timestamp
    frame[0]->poincloud of 3 channel
    type(frame[0])-> dtype=[('f0', '<f4'), ('f1', '<f4'), ('f2', '<f4')])
    frame[1]->timestamp


'''

def dump_to_pickle(depthFilePth):
    with lock:
        arrays = []
        while not buffer.empty():
            arrays.append(buffer.get())
        with open(depthFilePth, 'ab') as f:  # Append mode
            pickle.dump(arrays, f)
            print(f"Dumped {len(arrays)} arrays to {depthFilePth}")

# Function to add data to buffer
def add_to_buffer(path, buffereDepthData):
    with lock:
        buffer.put(buffereDepthData)
        if buffer.full():
            # print(path)
            dump_thread = threading.Thread(target=dump_to_pickle, args=(path,))
            dump_thread.start()



def collect_depth_data(duration,filename):
    directory_path = os.path.join('./datasets/', 'depth_data')
    folName = "_".join(filename.split("_")[1:5])
    imageDirectory_path = os.path.join('./datasets/', 'image_data/',folName)
    full_path = os.path.join(directory_path, filename)
    print(full_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("depth_data directory is created")
    if not os.path.exists(imageDirectory_path):
        os.makedirs(imageDirectory_path)
        print("Image_data directory is created")
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"File {full_path} already existed. Overwriting...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        index = 0
        end_time = time.time() + duration
        # while(True):
        while(time.time() < end_time):
            frames = pipeline.wait_for_frames()
            
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            pc = rs.pointcloud()
            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)
            pointData = np.asanyarray(points.get_vertices())
            images_path = os.path.join(imageDirectory_path,str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f'))+".jpg")
            # print(images_path)
            cv2.imwrite(images_path, color_image)
            index+=1
            combinedPointcloudTime = [pointData,datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")]
            add_to_buffer(full_path,combinedPointcloudTime)

        if not buffer.empty():
            dump_thread = threading.Thread(target=dump_to_pickle, args=(full_path,))
            dump_thread.start()
            dump_thread.join()


    finally:
        pipeline.stop()
        time.sleep(0.02)


if __name__ == "__main__":
    n_frames = 500
    periodicity = 200
    depth_duration = (int(n_frames)+5)*int(periodicity) / 1000
    print(depth_duration)
    collect_depth_data(duration=depth_duration,filename="drone_2024-09-10_16_12_18_test_depth.pkl")