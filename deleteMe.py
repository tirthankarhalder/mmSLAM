import threading
import pickle
import queue
import time
import numpy as np

# Buffer and other global variables
buffer_size = 2
buffer = queue.Queue(maxsize=buffer_size)
lock = threading.Lock()

# Function to dump buffer to pickle file
def dump_to_pickle():
    with lock:
        arrays = []
        while not buffer.empty():
            arrays.append(buffer.get())
        with open('dumped_data.pkl', 'ab') as f:  # Append mode
            pickle.dump(arrays, f)
        print(f"Dumped {len(arrays)} arrays to dumped_data.pkl")

# Function to add data to buffer
def add_to_buffer(data):
    with lock:
        buffer.put(data)
        if buffer.full():
            dump_thread = threading.Thread(target=dump_to_pickle)
            dump_thread.start()

# Simulate the process of getting large arrays in a loop
for i in range(5):
    # Simulating a large array
    large_array = np.random.rand(100)  # Replace this with your actual large array
    print(f"Adding {large_array} to buffer.")
    add_to_buffer(large_array)
    # time.sleep(1)  # Simulate some processing time

# Ensure remaining data is dumped after the loop
if not buffer.empty():
    dump_thread = threading.Thread(target=dump_to_pickle)
    dump_thread.start()
    dump_thread.join()

print("All data has been dumped.")
