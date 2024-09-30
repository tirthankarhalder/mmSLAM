import board 
import adafruit_mpu6050
import threading
import os 
import time 
import struct 
import csv
from datetime import datetime
header = [
        "datetime",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz"
        
    ]
def collect_data(duration, filename,csvfilename):
    i2c = board.I2C()  # uses board.SCL and board.SDA
    mpu = adafruit_mpu6050.MPU6050(i2c)
    csvFileName = csvfilename
    directory_path = os.path.join('./datasets/', 'imu_data')
    full_path = os.path.join(directory_path, filename)
    csvFilePath = os.path.join(directory_path, csvFileName)
    print(full_path)    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("imu_data directory is created")
    
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"File {full_path} already existed. Overwriting...")
    with open(csvFilePath, "w") as f:
        csv.DictWriter(f, fieldnames=header).writeheader()

    with open(full_path, 'wb') as file:
        end_time = time.time() + duration
        # while(True):
        while(time.time() < end_time):
            ax = mpu.acceleration[0]
            ay = mpu.acceleration[1]
            az = mpu.acceleration[2]
            gx = mpu.gyro[0]
            gy = mpu.gyro[1]
            gz = mpu.gyro[2]

            imu_data = [ax,ay,az,gx,gy,gz] 
            timestamp = time.time()
            data_to_store = struct.pack('d' * 7, timestamp, *imu_data)
                
            file.write(data_to_store)

            #csv file save
            # dict_dumper = {'datetime': datetime.now()}
            # data = {
            #     "ax":mpu.acceleration[0],
            #     "ay":mpu.acceleration[1],
            #     "az":mpu.acceleration[2],
            #     "gx":mpu.gyro[0],
            #     "gy":mpu.gyro[1],
            #     "gz":mpu.gyro[2]
            # }
            
            # dict_dumper.update(data)
            # with open(csvFilePath, "a") as f:
            #     writer = csv.DictWriter(f, header)
            #     writer.writerow(dict_dumper)

            # Schedule the next collection in 0.02 seconds
            #threading.Timer(0.02, collect_and_store).start()
            time.sleep(0.02)
                
        #collect_and_store()
