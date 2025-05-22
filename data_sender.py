import cv2
import numpy as np
from pupil_labs.realtime_api.simple import discover_one_device

# Import new modules
import screen_processing
import gaze_sender_network
import ui_manager

def main():
    print("Attempting to discover Pupil Labs Neon device...")
    device = discover_one_device(max_search_duration_seconds=5)
    if device is None:
        print("Error: Could not find Pupil Labs Neon device. Exiting.")
        return
    print(f"Connected to device: {getattr(device, 'full_name', 'Pupil Labs Neon Device')}")

    # Initialize components from new modules
    gaze_sender = gaze_sender_network.GazeDataSender()
    window_name = "Live Video Feed + Gaze Sender"
    opencv_ui = ui_manager.UIManager(window_name)

    # Setup trackbars using UIManager and screen_processing callbacks/initial values
    initial_trackbar_params = {
        "Canny Thr1": screen_processing.canny_thr_1,
        "Canny Thr2": screen_processing.canny_thr_2,
        "Blur Kernel (0-4 -> 1-9)": screen_processing.blur_kernel_trackbar,
        "Approx Poly Eps (1-50->.01-.5)": screen_processing.approx_poly_epsilon_trackbar,
        "AR Tolerance (1-30->.01-.3)": screen_processing.aspect_ratio_tolerance_trackbar,
        "Min Area % (1-50->.1-5%)": screen_processing.min_area_percent_trackbar
    }
    trackbar_callbacks = {
        "Canny Thr1": screen_processing.on_canny_thr1_change,
        "Canny Thr2": screen_processing.on_canny_thr2_change,
        "Blur Kernel (0-4 -> 1-9)": screen_processing.on_blur_kernel_change,
        "Approx Poly Eps (1-50->.01-.5)": screen_processing.on_approx_poly_epsilon_change,
        "AR Tolerance (1-30->.01-.3)": screen_processing.on_aspect_ratio_tolerance_change,
        "Min Area % (1-50->.1-5%)": screen_processing.on_min_area_percent_change
    }
    opencv_ui.setup_trackbars(initial_trackbar_params, trackbar_callbacks)
    opencv_ui.show_instructions()

    H_matrix = None 

    try:
        while True:
            frame = device.receive_scene_video_frame()
            if frame is None:
                if opencv_ui.get_keypress(30) == ord('q'):
                    break
                continue

            scene_img = frame.bgr_pixels
            display_img = scene_img.copy()
            
            # Use screen_processing module for detection
            detected_corners = screen_processing.detect_screen_corners(scene_img)
            current_H_valid = False

            if detected_corners is not None:
                screen_coordinates_actual = np.array([[0,0],[1920,0],[1920,1080],[0,1080]], dtype=np.float32)
                H_matrix, mask = cv2.findHomography(detected_corners, screen_coordinates_actual, cv2.RANSAC, 5.0)
                if H_matrix is not None:
                    current_H_valid = True
            else:
                H_matrix = None # Ensure H_matrix is None if no corners detected

            # Use UIManager to draw detection info
            display_img = opencv_ui.draw_detection_info(display_img, detected_corners, current_H_valid)

            if current_H_valid and H_matrix is not None:
                gaze = device.receive_gaze_datum()
                if gaze is not None and gaze.worn:
                    gx_orig, gy_orig = gaze.x, gaze.y # Pupil Labs coordinates (normalized 0-1)
                    
                    # Transform gaze to screen coordinates
                    # Denominator for perspective transformation
                    denom = H_matrix[2,0]*gx_orig + H_matrix[2,1]*gy_orig + H_matrix[2,2]
                    if denom != 0: # Avoid division by zero
                        px = (H_matrix[0,0]*gx_orig + H_matrix[0,1]*gy_orig + H_matrix[0,2]) / denom
                        py = (H_matrix[1,0]*gx_orig + H_matrix[1,1]*gy_orig + H_matrix[1,2]) / denom
                        ts = gaze.timestamp_unix_ns
                        
                        # Use GazeDataSender to send data
                        gaze_sender.send_gaze_data(ts, gx_orig, gy_orig, px, py)
                        # print(f"Sent: px={px:.2f}, py={py:.2f}") # Moved to sender or keep for main debug
                    else:
                        # print("Denominator is zero, cannot transform gaze.") # Optional debug
                        pass
            
            opencv_ui.display_image(display_img)
            key = opencv_ui.get_keypress(30)
            if key == ord('q'):
                print("Quitting...")
                break
    
    except Exception as e:
        print(f"An error occurred in main loop: {e}")
    finally:
        print("Closing device, sender, and UI.")
        if device:
            device.close()
        gaze_sender.close()
        opencv_ui.destroy_windows()
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()