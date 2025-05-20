import struct
import pickle
import numpy as np
# import utils.configuration as cfg
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
from mmwave.dsp.doppler_processing import separate_tx
from mmwave.dsp.utils import Window

# def read8byte(x):
#     return struct.unpack('<hhhh', x)

def get_info(args):
    datasetCSVPath = "/".join(args.split("/")[:-2])
    dataset=pd.read_csv(datasetCSVPath + '/dataset.csv')
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


def run_data_read_only_sensor(args, info_dict):
    datasetCSVPath = "/".join(args.split("/")[:-1])

    filename = datasetCSVPath + '/'+info_dict["filename"][0]
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
    pc_tmp = np.zeros((pc_size, 7), dtype=np.float32)
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
        # det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)

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




def generate_pcd_time_openradar(filename, info_dict, fixedPoint = False,fixedPointVal=1000):
    range_resolution, bandwidth = dsp.range_resolution(cfg.NUM_RANGE_BINS)

    NUM_FRAMES = info_dict[' Nf'][0]
    with open(filename, 'rb') as ADCBinFile: 
            frames = np.frombuffer(ADCBinFile.read(cfg.FRAME_SIZE*4*NUM_FRAMES), dtype=np.uint16)
    all_data = frame_reshape(frames, NUM_FRAMES)
    num_vec, steering_vec = dsp.gen_steering_vec(cfg.ANGLE_RANGE, cfg.ANGLE_RES, cfg.VIRT_ANT)
    # tracker = EKF()
    pointcloud = []
    frameID = 0
    pcd_datas = []
    time_frames = []
    rangeResult = []
    dopplerResult = []
    rangeAzimuthzResult = []
    rawHeatmap = []
    powerProfileValues = []
    snrs = []
    start_time = filename.split('/')[-1].split('.')[0].split('drone_')[-1][:19]
    start_time_obj = datetime.strptime(start_time,'%Y-%m-%d_%H_%M_%S')
    for adc_data in all_data:
        print(f"Frame initialized: {frameID} {filename}")
        time_current = start_time_obj+timedelta(seconds=frameID*(info_dict["periodicity"][0])/1000)
        time_frames.append(time_current.strftime('%Y-%m-%d %H_%M_%S.%f'))
        frameID+=1
        radar_cube = dsp.range_processing(adc_data)
        # mean = radar_cube.mean(0)
        # radar_cube = radar_cube - mean  
        # rangeResult.append(radar_cube)
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
        dopplerResult.append(det_matrix)
        # (4) Object Detection
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)
        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                axis=0,
                                                                arr=fft2d_sum.T,
                                                                l_bound=1.5,
                                                                guard_len=4,
                                                                noise_len=16)

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                            axis=0,
                                                            arr=fft2d_sum,
                                                            l_bound=2.5,
                                                            guard_len=4,
                                                            noise_len=16)

        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snrs.append(snr)
        dtype_location = '(' + str(cfg.NUM_TX) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, cfg.NUM_DOPPLER_BINS, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, cfg.NUM_DOPPLER_BINS)
        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, cfg.ADC_SAMPLES, 0.5, range_resolution)

        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
        rangeAzimuthzResult.append(azimuthInput)

        x, y, z = dsp.naive_xyz(azimuthInput.T)
        # print(x,y,z)
        xyzVecN = np.zeros((3, x.shape[0]))
        xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
        xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
        xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']
        Psi, Theta, Ranges, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'],
                                                                    range_resolution, method='Bartlett')
        pointcloud.append(xyzVec)
        # if frameID==5:
        #     break
    return pointcloud, time_frames
            

def generate_pcd_time(filename, info_dict, fixedPoint = False,fixedPointVal=1000):
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
    rangeResult = []
    dopplerResult = []
    rangeAzimuthzResult = []
    rawHeatmap = []
    powerProfileValues = []
    start_time = filename.split('/')[-1].split('.')[0].split('drone_')[-1][:19]
    start_time_obj = datetime.strptime(start_time,'%Y-%m-%d_%H_%M_%S')
    for adc_data in all_data:
        time_current = start_time_obj+timedelta(seconds=frameID*(info_dict["periodicity"][0])/1000)
        time_frames.append(time_current.strftime('%Y-%m-%d %H_%M_%S.%f'))
        frameID+=1
        radar_cube = dsp.range_processing(adc_data)
        mean = radar_cube.mean(0)
        radar_cube = radar_cube - mean  
        rangeResult.append(radar_cube)
        separate_tx_radar_cube = separate_tx(radar_cube,3,1,0)
        doppler_fft_value = np.fft.fft(separate_tx_radar_cube, axis=0)
        doppler_fft_value = np.fft.fftshift(doppler_fft_value, axes=0)
        doppler_fft_value = np.abs(doppler_fft_value.sum(axis=1))
        doppler_fft_value_scaled = (doppler_fft_value-np.min(doppler_fft_value))/(np.max(doppler_fft_value)-np.min(doppler_fft_value))
        dopplerResult.append(doppler_fft_value_scaled)
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

        powerProfile = np.sum(np.abs(radar_cube) ** 2, axis=(0, 1))  
        powerProfile  = powerProfile[ranges]

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
        rangeAzimuthzResult.append([ranges,azimuths])
        # print(f"ranges: {ranges.shape}, azimuths: {azimuths.shape}, dopplers: {dopplers.shape}, snrs: {snrs.shape}, powerProfile: {powerProfile.shape}")
        
        # print(powerProfile)
        # powerProfileValues.append(powerProfile)

        # --- put into EKF
        tracker.update_point_cloud(ranges, azimuths, dopplers, snrs, powerProfile)
        targetDescr, tNum = tracker.step()
        # frame_pcd = np.zeros((len(tracker.point_cloud),6))
        frame_pcd = np.zeros((len(tracker.point_cloud),7))
        for point_cloud, idx in zip(tracker.point_cloud, range(len(tracker.point_cloud))):
            frame_pcd[idx,0] = -np.sin(point_cloud.angle) * point_cloud.range
            frame_pcd[idx,1] = np.cos(point_cloud.angle) * point_cloud.range
            frame_pcd[idx,2] = point_cloud.doppler 
            frame_pcd[idx,3] = point_cloud.snr
            frame_pcd[idx,4] = point_cloud.range
            frame_pcd[idx,5] = point_cloud.angle
            frame_pcd[idx,6] = point_cloud.power

        if fixedPoint:
            frame_pcd = reg_data(frame_pcd,fixedPointVal)
            
        pcd_datas.append(frame_pcd)
        rawHeatmap.append(heatmap_log)
    pointcloud = np.array(pcd_datas)
   
    return pointcloud, time_frames, dopplerResult, rangeResult, rangeAzimuthzResult,rawHeatmap #, powerProfileValues