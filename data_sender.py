import cv2
import numpy as np
import socket
import struct
from pupil_labs.realtime_api.simple import discover_one_device

# --- Tunable Parameters (Global) ---
# Values from your screen_detector.py
canny_thr_1 = 5
canny_thr_2 = 25
blur_kernel_trackbar = 4
approx_poly_epsilon_trackbar = 14
aspect_ratio_tolerance_trackbar = 26
min_area_percent_trackbar = 17

# --- Callback functions for trackbars ---
def on_canny_thr1_change(val):
    global canny_thr_1
    canny_thr_1 = val

def on_canny_thr2_change(val):
    global canny_thr_2
    canny_thr_2 = val

def on_blur_kernel_change(val):
    global blur_kernel_trackbar
    blur_kernel_trackbar = val

def on_approx_poly_epsilon_change(val):
    global approx_poly_epsilon_trackbar
    approx_poly_epsilon_trackbar = val

def on_aspect_ratio_tolerance_change(val):
    global aspect_ratio_tolerance_trackbar
    aspect_ratio_tolerance_trackbar = val

def on_min_area_percent_change(val):
    global min_area_percent_trackbar
    min_area_percent_trackbar = val

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff_yx = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff_yx)] # Top-right
    rect[3] = pts[np.argmax(diff_yx)] # Bottom-left
    return rect

def detect_screen_corners(image):
    global canny_thr_1, canny_thr_2, blur_kernel_trackbar, \
           approx_poly_epsilon_trackbar, aspect_ratio_tolerance_trackbar, \
           min_area_percent_trackbar

    current_blur_kernel_size = 2 * blur_kernel_trackbar + 1
    if current_blur_kernel_size < 1: current_blur_kernel_size = 1

    current_approx_poly_epsilon = (approx_poly_epsilon_trackbar / 100.0)
    if current_approx_poly_epsilon < 0.01: current_approx_poly_epsilon = 0.01

    current_aspect_ratio_tolerance = aspect_ratio_tolerance_trackbar / 100.0
    if current_aspect_ratio_tolerance < 0.01: current_aspect_ratio_tolerance = 0.01
    
    current_min_area_factor = min_area_percent_trackbar / 1000.0
    if current_min_area_factor < 0.0001: current_min_area_factor = 0.0001

    target_aspect_ratio = 1920.0 / 1080.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (current_blur_kernel_size, current_blur_kernel_size), 0)
    edged = cv2.Canny(blurred, canny_thr_1, canny_thr_2)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None 

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    found_corners = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, current_approx_poly_epsilon * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2)
            ordered_corners = order_points(points.astype(np.float32))
            tl, tr, br, bl = ordered_corners
            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            avg_width = (width_top + width_bottom) / 2.0
            height_left = np.linalg.norm(bl - tl)
            height_right = np.linalg.norm(br - tr)
            avg_height = (height_left + height_right) / 2.0

            if avg_height == 0 or avg_width == 0:
                continue

            aspect_ratio_detected = avg_width / avg_height
            if abs(aspect_ratio_detected - target_aspect_ratio) <= current_aspect_ratio_tolerance * target_aspect_ratio:
                min_area_val = current_min_area_factor * image.shape[0] * image.shape[1]
                if cv2.contourArea(approx) < min_area_val:
                    continue
                found_corners = ordered_corners.astype(np.float32)
                break
    return found_corners


def main():
    print("Attempting to discover Pupil Labs Neon device...")
    device = discover_one_device(max_search_duration_seconds=5)
    if device is None:
        print("Error: Could not find Pupil Labs Neon device. Exiting.")
        return
    print(f"Connected to device: {getattr(device, 'full_name', 'Pupil Labs Neon Device')}")

    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cv2.namedWindow("Live Video Feed + Gaze Sender")
    cv2.createTrackbar("Canny Thr1", "Live Video Feed + Gaze Sender", canny_thr_1, 255, on_canny_thr1_change)
    cv2.createTrackbar("Canny Thr2", "Live Video Feed + Gaze Sender", canny_thr_2, 255, on_canny_thr2_change)
    cv2.createTrackbar("Blur Kernel (0-4 -> 1-9)", "Live Video Feed + Gaze Sender", blur_kernel_trackbar, 4, on_blur_kernel_change)
    cv2.createTrackbar("Approx Poly Eps (1-50->.01-.5)", "Live Video Feed + Gaze Sender", approx_poly_epsilon_trackbar, 50, on_approx_poly_epsilon_change)
    cv2.createTrackbar("AR Tolerance (1-30->.01-.3)", "Live Video Feed + Gaze Sender", aspect_ratio_tolerance_trackbar, 30, on_aspect_ratio_tolerance_change)
    cv2.createTrackbar("Min Area % (1-50->.1-5%)", "Live Video Feed + Gaze Sender", min_area_percent_trackbar, 50, on_min_area_percent_change)

    print("\nPress 'q' to quit.")
    print("Adjust trackbars to tune detection parameters.")
    print("Gaze data will be sent only when the screen is detected and homography is computed.")

    H_matrix = None 

    try:
        while True:
            frame = device.receive_scene_video_frame()
            if frame is None:
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                continue

            scene_img = frame.bgr_pixels
            display_img = scene_img.copy()
            
            detected_corners = detect_screen_corners(scene_img)
            current_H_valid = False

            if detected_corners is not None:
                cv2.polylines(display_img, [detected_corners.astype(np.int32)], True, (0, 255, 0), 2)
                corner_labels = ["TL", "TR", "BR", "BL"]
                for i, point in enumerate(detected_corners):
                    pt_int = tuple(point.astype(int))
                    cv2.circle(display_img, pt_int, 5, (0, 0, 255), -1)
                    cv2.putText(display_img, corner_labels[i], (pt_int[0] + 10, pt_int[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                screen_coordinates_actual = np.array([[0,0],[1920,0],[1920,1080],[0,1080]], dtype=np.float32)
                H_matrix, mask = cv2.findHomography(detected_corners, screen_coordinates_actual, cv2.RANSAC, 5.0)

                if H_matrix is not None:
                    current_H_valid = True
                    cv2.putText(display_img, "Screen Detected. Sending Gaze.", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, "Screen Detected. Homography Failed.", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # orange
            else:
                H_matrix = None 
                cv2.putText(display_img, "No screen detected. Adjust parameters.", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if current_H_valid and H_matrix is not None:
                gaze = device.receive_gaze_datum()
                if gaze is not None and gaze.worn:
                    gx, gy = gaze.x, gaze.y
                    denom = H_matrix[2,0]*gx + H_matrix[2,1]*gy + H_matrix[2,2]
                    if denom != 0:
                        px = (H_matrix[0,0]*gx + H_matrix[0,1]*gy + H_matrix[0,2]) / denom
                        py = (H_matrix[1,0]*gx + H_matrix[1,1]*gy + H_matrix[1,2]) / denom
                        ts = gaze.timestamp_unix_ns
                        packet = struct.pack('<dffff', ts, gx, gy, float(px), float(py))
                        sock.sendto(packet, (UDP_IP, UDP_PORT))
                        print(f"Sent: px={px:.2f}, py={py:.2f}") # debugging
                    else:
                        pass
                

            cv2.imshow("Live Video Feed + Gaze Sender", display_img)
            key = cv2.waitKey(30) & 0xFF # Increased wait time for smoother UI
            if key == ord('q'):
                print("Quitting...")
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing device, socket, and windows.")
        if device:
            device.close()
        if sock:
            sock.close()
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()