#define CHIRPS 128
#define RX 4
#define TX 3
#define SAMPLES 256
#define IQ 2
#define BYTES 2
#define BYTES_IN_PACKET 1456
#define BYTES_IN_FRAME (CHIRPS * RX * TX * IQ * SAMPLES * BYTES)
#define BYTES_IN_FRAME_CLIPPED ((BYTES_IN_FRAME / BYTES_IN_PACKET) * BYTES_IN_PACKET)
#define PACKETS_IN_FRAME (BYTES_IN_FRAME / BYTES_IN_PACKET)
#define PACKETS_IN_FRAME_CLIPPED (BYTES_IN_FRAME / BYTES_IN_PACKET)
#define UINT16_IN_PACKET (BYTES_IN_PACKET / 2)
#define UINT16_IN_FRAME (BYTES_IN_FRAME / 2)

#define STATIC_IP "192.168.33.30"
#define DATA_PORT 4098

void get_sensor_data(const char* filename, int num_frames);
