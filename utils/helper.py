import struct
import pickle
import numpy as np
import utils.configuration as cfg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import subprocess
import statistics
from scipy.signal import find_peaks
from datetime import datetime,timedelta
from mmwave.dataloader import DCA1000
import mmwave.dsp as dsp
from mmwave.tracking import EKF


# def read8byte(x):
#     return struct.unpack('<hhhh', x)

def get_info(args):
    dataset=pd.read_csv('./datasets/dataset.csv')
    # print(dataset)
    file_name=args
    file_name = file_name.split("/")[3]
    filtered_row=dataset[dataset['filename']==file_name]
    info_dict={}
    for col in dataset.columns:
        info_dict[col]=filtered_row[col].values
    if len(info_dict['filename'])==0:
        print('Oops! File not found in database. Cross check the file name')
    return info_dict


def print_info(info_dict):
    print('***************************************************************')
    print('Printing the file profile')
    print(f'--filename: {"only_sensor"+info_dict["filename"][0]}')
    print(f'--Length(L in cm): {info_dict[" L"][0]}')
    print(f'--Radial_Length(R in cm): {info_dict[" R"][0]}')
    print(f'--PWM Value: {info_dict[" PWM"][0]}')
    print(f'--A brief desciption: {info_dict[" Description"][0]}')
    print('***************************************************************')


def run_data_read_only_sensor(info_dict):
    filename = './datasets/radar_data/'+info_dict["filename"][0]
    # filename = info_dict["filename"][0]
    command =f'python data_read_only_sensor.py {filename} {info_dict[" Nf"][0]}'
    try:
        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(e)
    stdout = process.stdout
    stderr = process.stderr

def call_destructor(info_dict):
    file_name="datasets/only_sensor"+info_dict["filename"][0]
    command =f'rm {file_name}'
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = process.stdout
    stderr = process.stderr


def frame_reshape(frames, NUM_FRAMES):
    try:
        adc_data = frames.reshape(NUM_FRAMES, -1)
    except ValueError as e:
        return None
    all_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=cfg.LOOPS_PER_FRAME*cfg.NUM_TX, num_rx=cfg.NUM_RX, num_samples=cfg.ADC_SAMPLES)
    return all_data

def reg_data(data, pc_size):  #
    pc_tmp = np.zeros((pc_size, 6), dtype=np.float32)
    pc_no = data.shape[0]
    if pc_no < pc_size:
        fill_list = np.random.choice(pc_size, size=pc_no, replace=False)
        fill_set = set(fill_list)
        pc_tmp[fill_list] = data
        dupl_list = [x for x in range(pc_size) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no, size=len(dupl_list), replace=True)
        pc_tmp[dupl_list] = data[dupl_pc]
    else:
        pc_list = np.random.choice(pc_no, size=pc_size, replace=False)
        pc_tmp = data[pc_list]
    return pc_tmp

def generate_pcd(filename, info_dict):
    NUM_FRAMES = info_dict[' Nf'][0]
    with open(filename, 'rb') as ADCBinFile: 
        frames = np.frombuffer(ADCBinFile.read(cfg.FRAME_SIZE*4*NUM_FRAMES), dtype=np.uint16)
    all_data = frame_reshape(frames, NUM_FRAMES)
    range_azimuth = np.zeros((cfg.ANGLE_BINS, cfg.ADC_SAMPLES))
    num_vec, steering_vec = dsp.gen_steering_vec(cfg.ANGLE_RANGE, cfg.ANGLE_RES, cfg.VIRT_ANT)
    tracker = EKF()
    count = 0
    pcd_datas = []
    for adc_data in all_data:
        count+=1
        radar_cube = dsp.range_processing(adc_data)
        mean = radar_cube.mean(0)                 
        radar_cube = radar_cube - mean  
        # --- capon beamforming
        beamWeights   = np.zeros((cfg.VIRT_ANT, cfg.ADC_SAMPLES), dtype=np.complex128)
        radar_cube = np.concatenate((radar_cube[0::3,...], radar_cube[1::3,...], radar_cube[2::3,...]), axis=1)
        # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
        # has doppler at the last dimension.
        for i in range(cfg.ADC_SAMPLES):
            range_azimuth[:,i], beamWeights[:,i] = dsp.aoa_capon(radar_cube[:, :, i].T, steering_vec, magnitude=True)
        
        """ 3 (Object Detection) """
        heatmap_log = np.log2(range_azimuth)
        
        # --- cfar in azimuth direction
        first_pass, _ = np.apply_along_axis(func1d=dsp.ca_,
                                            axis=0,
                                            arr=heatmap_log,
                                            l_bound=1.5,
                                            guard_len=4,
                                            noise_len=16)
        
        # --- cfar in range direction
        second_pass, noise_floor = np.apply_along_axis(func1d=dsp.ca_,
                                                    axis=0,
                                                    arr=heatmap_log.T,
                                                    l_bound=2.5,
                                                    guard_len=4,
                                                    noise_len=16)

        # --- classify peaks and caclulate snrs
        noise_floor = noise_floor.T
        first_pass = (heatmap_log > first_pass)
        second_pass = (heatmap_log > second_pass.T)
        peaks = (first_pass & second_pass)
        peaks[:cfg.SKIP_SIZE, :] = 0
        peaks[-cfg.SKIP_SIZE:, :] = 0
        peaks[:, :cfg.SKIP_SIZE] = 0
        peaks[:, -cfg.SKIP_SIZE:] = 0
        pairs = np.argwhere(peaks)
        azimuths, ranges = pairs.T
        snrs = heatmap_log[pairs[:,0], pairs[:,1]] - noise_floor[pairs[:,0], pairs[:,1]]

        """ 4 (Doppler Estimation) """

        # --- get peak indices
        # beamWeights should be selected based on the range indices from CFAR.
        dopplerFFTInput = radar_cube[:, :, ranges]
        beamWeights  = beamWeights[:, ranges]

        # --- estimate doppler values
        # For each detected object and for each chirp combine the signals from 4 Rx, i.e.
        # For each detected object, matmul (numChirpsPerFrame, numRxAnt) with (numRxAnt) to (numChirpsPerFrame)
        dopplerFFTInput = np.einsum('ijk,jk->ik', dopplerFFTInput, beamWeights)
        if not dopplerFFTInput.shape[-1]:
            continue
        dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
        dopplerEst = np.argmax(dopplerEst, axis=0)
        dopplerEst[dopplerEst[:]>=cfg.LOOPS_PER_FRAME/2] -= cfg.LOOPS_PER_FRAME
        
        """ 5 (Extended Kalman Filter) """

        # --- convert bins to units
        ranges = ranges * cfg.RANGE_RESOLUTION
        azimuths = (azimuths - (cfg.NUM_ANGLE_BINS // 2)) * (np.pi / 180)
        dopplers = dopplerEst * cfg.DOPPLER_RESOLUTION
        snrs = snrs
        
        # --- put into EKF
        tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
        targetDescr, tNum = tracker.step()
        frame_pcd = np.zeros((len(tracker.point_cloud),6))
        for point_cloud, idx in zip(tracker.point_cloud, range(len(tracker.point_cloud))):
            frame_pcd[idx,0] = -np.sin(point_cloud.angle) * point_cloud.range
            frame_pcd[idx,1] = np.cos(point_cloud.angle) * point_cloud.range
            frame_pcd[idx,2] = point_cloud.doppler 
            frame_pcd[idx,3] = point_cloud.snr
            frame_pcd[idx,4] = point_cloud.range
            frame_pcd[idx,5] = point_cloud.angle
        pcd_datas.append(frame_pcd)
    return np.array(pcd_datas)




def generate_pcd_time(filename, info_dict, fixedPoint = False):
    NUM_FRAMES = info_dict[' Nf'][0]
    with open(filename, 'rb') as ADCBinFile: 
        frames = np.frombuffer(ADCBinFile.read(cfg.FRAME_SIZE*4*NUM_FRAMES), dtype=np.uint16)
    all_data = frame_reshape(frames, NUM_FRAMES)
    range_azimuth = np.zeros((cfg.ANGLE_BINS, cfg.ADC_SAMPLES))
    num_vec, steering_vec = dsp.gen_steering_vec(cfg.ANGLE_RANGE, cfg.ANGLE_RES, cfg.VIRT_ANT)
    tracker = EKF()
    frameID = 0
    pcd_datas = []
    time_frames = []
    start_time = filename.split('/')[-1].split('.')[0].split('drone_')[-1][:19]
    start_time_obj = datetime.strptime(start_time,'%Y-%m-%d_%H_%M_%S')
    for adc_data in all_data:
        time_current = start_time_obj+timedelta(seconds=frameID*(info_dict["periodicity"][0])/1000)
        time_frames.append(time_current.strftime('%Y-%m-%d %H_%M_%S.%f'))
        frameID+=1
        radar_cube = dsp.range_processing(adc_data)
        mean = radar_cube.mean(0)                 
        radar_cube = radar_cube - mean  
        # --- capon beamforming
        beamWeights   = np.zeros((cfg.VIRT_ANT, cfg.ADC_SAMPLES), dtype=np.complex128)
        radar_cube = np.concatenate((radar_cube[0::3,...], radar_cube[1::3,...], radar_cube[2::3,...]), axis=1)
        # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
        # has doppler at the last dimension.
        for i in range(cfg.ADC_SAMPLES):
            range_azimuth[:,i], beamWeights[:,i] = dsp.aoa_capon(radar_cube[:, :, i].T, steering_vec, magnitude=True)
        
        """ 3 (Object Detection) """
        heatmap_log = np.log2(range_azimuth)
        
        # --- cfar in azimuth direction
        first_pass, _ = np.apply_along_axis(func1d=dsp.ca_,
                                            axis=0,
                                            arr=heatmap_log,
                                            l_bound=1.5,
                                            guard_len=4,
                                            noise_len=16)
        
        # --- cfar in range direction
        second_pass, noise_floor = np.apply_along_axis(func1d=dsp.ca_,
                                                    axis=0,
                                                    arr=heatmap_log.T,
                                                    l_bound=2.5,
                                                    guard_len=4,
                                                    noise_len=16)

        # --- classify peaks and caclulate snrs
        noise_floor = noise_floor.T
        first_pass = (heatmap_log > first_pass)
        second_pass = (heatmap_log > second_pass.T)
        peaks = (first_pass & second_pass)
        peaks[:cfg.SKIP_SIZE, :] = 0
        peaks[-cfg.SKIP_SIZE:, :] = 0
        peaks[:, :cfg.SKIP_SIZE] = 0
        peaks[:, -cfg.SKIP_SIZE:] = 0
        pairs = np.argwhere(peaks)
        azimuths, ranges = pairs.T
        snrs = heatmap_log[pairs[:,0], pairs[:,1]] - noise_floor[pairs[:,0], pairs[:,1]]

        """ 4 (Doppler Estimation) """

        # --- get peak indices
        # beamWeights should be selected based on the range indices from CFAR.
        dopplerFFTInput = radar_cube[:, :, ranges]
        beamWeights  = beamWeights[:, ranges]

        # --- estimate doppler values
        # For each detected object and for each chirp combine the signals from 4 Rx, i.e.
        # For each detected object, matmul (numChirpsPerFrame, numRxAnt) with (numRxAnt) to (numChirpsPerFrame)
        dopplerFFTInput = np.einsum('ijk,jk->ik', dopplerFFTInput, beamWeights)
        if not dopplerFFTInput.shape[-1]:
            continue
        dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
        dopplerEst = np.argmax(dopplerEst, axis=0)
        dopplerEst[dopplerEst[:]>=cfg.LOOPS_PER_FRAME/2] -= cfg.LOOPS_PER_FRAME
        
        """ 5 (Extended Kalman Filter) """

        # --- convert bins to units
        ranges = ranges * cfg.RANGE_RESOLUTION
        azimuths = (azimuths - (cfg.NUM_ANGLE_BINS // 2)) * (np.pi / 180)
        dopplers = dopplerEst * cfg.DOPPLER_RESOLUTION
        snrs = snrs
        
        # --- put into EKF
        tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
        targetDescr, tNum = tracker.step()
        frame_pcd = np.zeros((len(tracker.point_cloud),6))
        for point_cloud, idx in zip(tracker.point_cloud, range(len(tracker.point_cloud))):
            frame_pcd[idx,0] = -np.sin(point_cloud.angle) * point_cloud.range
            frame_pcd[idx,1] = np.cos(point_cloud.angle) * point_cloud.range
            frame_pcd[idx,2] = point_cloud.doppler 
            frame_pcd[idx,3] = point_cloud.snr
            frame_pcd[idx,4] = point_cloud.range
            frame_pcd[idx,5] = point_cloud.angle

        if fixedPoint:
            frame_pcd = reg_data(frame_pcd,1000)
            
        pcd_datas.append(frame_pcd)

    pointcloud = np.array(pcd_datas)
    
    return pointcloud, time_frames