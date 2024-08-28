# ## setup logging
# import logging
# logging.basicConfig(level = logging.INFO)

# ## import the package
# import pyrealsense as pyrs

# ## start the service - also available as context manager
# serv = pyrs.Service()

# ## create a device from device id and streams of interest
# cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = 60)])

# ## retrieve 60 frames of data
# for _ in range(60):
#     cam.wait_for_frames()
#     print(cam.color)

# ## stop camera and service
# cam.stop()
# serv.stop()
import pyrealsense2 as rs



pipe = rs.pipeline()
profile = pipe.start()
try:
  for i in range(0, 100):
    frames = pipe.wait_for_frames()
    for f in frames:
      print(f.profile)
finally:
    pipe.stop()