from mpu6050 import mpu6050
import time

# Initialize MPU-6500 (address is typically 0x68)
sensor = mpu6050(0x68)

while True:
    # Read accelerometer and gyroscope data
    accelerometer_data = sensor.get_accel_data()
    gyroscope_data = sensor.get_gyro_data()

    # Print accelerometer and gyroscope data
    print(f"Accelerometer: {accelerometer_data}")
    print(f"Gyroscope: {gyroscope_data}")

    # Delay to control the data reading frequency
    time.sleep(1)
