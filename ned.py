#!/usr/bin/env python3

import asyncio
from mavsdk import System
import csv
from datetime import datetime

header = [
    "datetime",
    "time_usec",
    "x_m",
    "y_m",
    "z_m",
    "x_m_s",
    "y_m_s",
    "z_m_s",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
    "roll_rad_s",
    "pitch_rad_s",
    "yaw_rad_s"

    
]
now = datetime.now()
date_string = str(now.strftime('%Y-%m-%d_%H_%M_%S'))
filepath = "home/cm/mmSLAM/telemetry_data/" + date_string + "_telemtry.csv"  
with open(filepath, "w") as f:
    csv.DictWriter(f, fieldnames=header).writeheader()


async def print_status_text():
    drone = System()
    await drone.connect(system_address="serial:///dev/serial0:921600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    # async for ned in drone.telemetry.positionbody():
    #     print("NED:", ned)


    async for odom in drone.telemetry.odometry():
        print("Odometry_time_usec:", odom.Odometry.time_usec)
        dict_dumper = {'datetime': datetime.now()}
        data = {
            "time_usec":odom.Odometry.time_usec,
            "x_m": odom.position_body.x_m,
            "y_m": odom.position_body.y_m,
            "z_m":odom.position_body.z_m,
            "x_m_s":odom.velocity_body.x_m_s,
            "y_m_s":odom.velocity_body.y_m_s,
            "z_m_s":odom.velocity_body.z_m_s,
            "q_w":odom.q.w,
            "q_x":odom.q.x,
            "q_y":odom.q.y,
            "q_z":odom.q.z,
            "roll_rad_s":odom.angular_velocity_body.roll_rad_s,
            "pitch_rad_s":odom.angular_velocity_body.pitch_rad_s,
            "yaw_rad_s":odom.angular_velocity_body.yaw_rad_s
        }
        
        dict_dumper.update(data)
        with open(filepath, "a") as f:
            writer = csv.DictWriter(f, header)
            writer.writerow(dict_dumper)

if __name__ == "__main__":
    asyncio.run(print_status_text())
