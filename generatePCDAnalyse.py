from helper import *    
import seaborn as sns


def pointcloud_openradar(file_name):
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = './datasets/radar_data/only_sensor_' + info_dict['filename'][0]
    pcd_data, time = generate_pcd_time(bin_filename, info_dict,fixedPoint=True)
    # print(pcd_data.shape)
    return pcd_data, time

if __name__ == "__main__":
    # gen=point_cloud_frames(file_name ='./datasets/radar_data/drone_2024-09-10_16_12_18_test.bin')
    gen, timestamps = pointcloud_openradar(file_name ='./datasets/radar_data/drone_2024-09-10_16_12_18_test.bin')
    print("timestamps.shape: ",len(timestamps))
    total_data = []
    total_ids = []
    total_frames=0
    first_frame = True
    initial_coordinates = {}
    current_cluster = {}
    points = []
    prev_point = np.array([0,0])
    frameID = 1
    visualization = True
    for pointcloud in gen:
        # print(pointcloud.shape)
        # print(pointcloud)
        # break
        if frameID == 1:
            print(pointcloud.shape,frameID)
            points = pointcloud[:,:3]
            print(points.shape,frameID)
            if visualization:
                sns.set(style="whitegrid")
                fig = plt.figure(figsize=(12,7))
                ax = fig.add_subplot(111,projection='3d')
                img = ax.scatter(points[:,0], points[:,1], points[:,2], cmap="jet",marker='o')
                fig.colorbar(img)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                plt.savefig("./visualization/radar/OpenRadarFixedPointPCD.png")
                # plt.show()
                plt.close()
            break
        frameID+=1