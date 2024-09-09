import board 
import adafruit_mpu6050


i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c,0x68)


while True:
    ax = mpu.acceleration[0]
    ay = mpu.acceleration[1]
    az = mpu.acceleration[2]
    gx = mpu.gyro[0]
    gy = mpu.gyro[1]
    gz = mpu.gyro[2]
    print("ax: {}, ay: {},az: {},gx: {},gy: {},gz: {}".format(ax,ay,az,gx,gy,gz))