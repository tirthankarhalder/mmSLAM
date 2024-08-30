import threading
import os 
import time 
import struct 
import csv
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
header = [
        "datetime",
        "frame_number",
        "x",
        "y",
        "z"
        
    ]

def collect_depth_data(duration,filename):
    # Construct the full path with the desired directory
    directory_path = os.path.join('./', 'depth')
    full_path = os.path.join(directory_path, filename)
    print(full_path)    
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("depth_data directory is created")
    
    # Check if the file exists, delete it if it does
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"File {full_path} already existed. Overwriting...")
   
    # filename = "depth.csv"  
    with open(full_path, "w") as f:
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
	
        end_time = time.time() + duration
        while(time.time() < end_time):
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

            #check properties

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
        time.sleep(0.02)

        # Close OpenCV windows
        # cv2.destroyAllWindows()
    # Open the file in binary write mode
    # with open(full_path, 'wb') as file:
    #     end_time = time.time() + duration
    #     while(time.time() < end_time):
    #         # Simulating the collection of IMU data (6 values) and a timestamp
    #         ax = mpu.acceleration[0]
    #         ay = mpu.acceleration[1]
    #         az = mpu.acceleration[2]
    #         gx = mpu.gyro[0]
    #         gy = mpu.gyro[1]
    #         gz = mpu.gyro[2]

    #         imu_data = [ax,ay,az,gx,gy,gz] # Placeholder for actual IMU data collection
    #         timestamp = time.time()
    #         data_to_store = struct.pack('d' * 7, timestamp, *imu_data)
                
    #         # Write the packed data to the file
    #         file.write(data_to_store)
                
    #         # Schedule the next collection in 0.02 seconds
    #         #threading.Timer(0.02, collect_and_store).start()
    #         time.sleep(0.02)
                
    #     # Start the first data collection
    #     #collect_and_store()


