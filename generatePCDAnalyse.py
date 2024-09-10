from helper import *    
import seaborn as sns


def point_cloud_frames(file_name = None):
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = './datasets/radar_data/only_sensor_' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()
    velocities = []
    pcds = []
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        if frame_no == 5:
            range_heatmap = np.sum(np.abs(rangeResult), axis=(0,1))
            print("range_heatmap.shape: ", range_heatmap.shape)
            sns.heatmap(range_heatmap)
            plt.savefig('range.png')
        
        dopplerResult = dopplerFFT(rangeResult, frameConfig)
        pointCloud = frame2pointcloud(dopplerResult, pointCloudProcessCFG)
        pcds.append(pointCloud)
    return pcds
        
gen=point_cloud_frames(file_name ='./datasets/radar_data/drone_2024-09-10_16_12_18_test.bin')
total_data = []
total_ids = []
total_frames=0
first_frame = True
initial_coordinates = {}
current_cluster = {}
points = []
prev_point = np.array([0,0])
for pointcloud in gen:
    print(pointcloud.shape)
    print(pointcloud)
    break
    