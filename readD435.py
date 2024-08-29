import pyrealsense2 as rs
import numpy as np
#import open3d as o3d
import cv2
#csv writer
import csv
from datetime import datetime
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
