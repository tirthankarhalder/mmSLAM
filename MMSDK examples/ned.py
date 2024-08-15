#!/usr/bin/env python3

import asyncio
from mavsdk import System


async def print_status_text():
    drone = System()
    await drone.connect(system_address="serial:///dev/serial0:921600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    async for ned in drone.telemetry.positionbody():
        print("NED:", ned)


    async for odom in drone.telemetry.odometry():
        print("Odometry:", odom)
if __name__ == "__main__":
    asyncio.run(print_status_text())
