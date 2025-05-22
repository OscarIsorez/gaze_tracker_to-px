\
import socket
import struct

class GazeDataSender:
    def __init__(self, udp_ip="127.0.0.1", udp_port=5005):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"GazeDataSender initialized. Will send to {self.udp_ip}:{self.udp_port}")

    def send_gaze_data(self, timestamp_unix_ns, gaze_x_original, gaze_y_original, gaze_x_transformed, gaze_y_transformed):
        """Packs and sends gaze data via UDP."""
        try:
            # <dffff means: little-endian, double, float, float, float, float
            packet = struct.pack('<dffff', 
                                 timestamp_unix_ns, 
                                 gaze_x_original, 
                                 gaze_y_original, 
                                 float(gaze_x_transformed), 
                                 float(gaze_y_transformed))
            self.sock.sendto(packet, (self.udp_ip, self.udp_port))
            print(f"Sent: px={gaze_x_transformed:.2f}, py={gaze_y_transformed:.2f}") # Optional: for debugging
        except Exception as e:
            print(f"Error sending gaze data: {e}")

    def close(self):
        """Closes the UDP socket."""
        if self.sock:
            self.sock.close()
            print("GazeDataSender socket closed.")
