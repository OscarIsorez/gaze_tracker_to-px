import cv2
import numpy as np
from pupil_labs.realtime_api.simple import discover_one_device

# --- Tunable Parameters (Global) ---
# Canny Edge Detection
canny_thr_1 = 5  # Initial value
canny_thr_2 = 25 # Initial value
# Gaussian Blur
blur_kernel_trackbar = 4  # 0:1x1, 1:3x3, 2:5x5, 3:7x7, 4:9x9 (Initial: 5x5)
# Contour Approximation
approx_poly_epsilon_trackbar = 14  # 1-50, maps to 0.01-0.50 for approxPolyDP (Changed from 2 to 3)
# Aspect Ratio
aspect_ratio_tolerance_trackbar = 26  # 1-30, maps to 0.01-0.30 tolerance (Changed from 15 to 20)
# Minimum Area
min_area_percent_trackbar =17  # 1-50, maps to 0.1% to 5.0% of image area (Changed from 10 to 5)

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

# Helper function to order points of a quadrilateral
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # For top-right and bottom-left, we need to look at the difference (y-x) or (x-y)
    # Using y - x: top-right has min diff, bottom-left has max diff
    diff_yx = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff_yx)] # Top-right
    rect[3] = pts[np.argmax(diff_yx)] # Bottom-left
    
    # Check if this order is correct by visual inspection or known good points
    # If TL and BR are correct, but TR and BL are swapped, the diff logic might need adjustment
    # e.g. if TR should have max (x-y) and BL min (x-y)
    # diff_xy = pts[:,0] - pts[:,1]
    # rect[1] = pts[np.argmax(diff_xy)]
    # rect[3] = pts[np.argmin(diff_xy)]
    return rect

# Function to detect screen corners using global tunable parameters
def detect_screen_corners_tuned(image):
    global canny_thr_1, canny_thr_2, blur_kernel_trackbar, \
           approx_poly_epsilon_trackbar, aspect_ratio_tolerance_trackbar, \
           min_area_percent_trackbar

    # Derive actual parameters from trackbar values
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
        return None, edged

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    found_corners = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, current_approx_poly_epsilon * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2)
            
            # Ensure points are float32 for order_points if it expects that
            ordered_corners = order_points(points.astype(np.float32))

            tl, tr, br, bl = ordered_corners
            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            avg_width = (width_top + width_bottom) / 2.0
            
            height_left = np.linalg.norm(tl - bl) # Should be tl-bl or bl-tl
            height_right = np.linalg.norm(tr - br) # Should be tr-br or br-tr
            avg_height = (height_left + height_right) / 2.0

            if avg_height == 0 or avg_width == 0: # Avoid division by zero
                continue

            aspect_ratio_detected = avg_width / avg_height
            
            if abs(aspect_ratio_detected - target_aspect_ratio) <= current_aspect_ratio_tolerance * target_aspect_ratio:
                min_area_val = current_min_area_factor * image.shape[0] * image.shape[1]
                if cv2.contourArea(approx) < min_area_val:
                    continue
                
                found_corners = ordered_corners.astype(np.float32)
                break 
    
    return found_corners, edged

def main():
    print("Attempting to discover Pupil Labs Neon device...")
    device = discover_one_device(max_search_duration_seconds=5)

    if device is None:
        print("Error: Could not find Pupil Labs Neon device. Exiting.")
        return

    print(f"Connected to device: {getattr(device, 'full_name', device.full_name)}") 

    cv2.namedWindow("Live Video Feed")
    # cv2.namedWindow("Canny Edges Preview") # Removed this line

    # Create trackbars
    cv2.createTrackbar("Canny Thr1", "Live Video Feed", canny_thr_1, 255, on_canny_thr1_change)
    cv2.createTrackbar("Canny Thr2", "Live Video Feed", canny_thr_2, 255, on_canny_thr2_change)
    cv2.createTrackbar("Blur Kernel (0-4 -> 1-9)", "Live Video Feed", blur_kernel_trackbar, 4, on_blur_kernel_change)
    cv2.createTrackbar("Approx Poly Eps (1-50->.01-.5)", "Live Video Feed", approx_poly_epsilon_trackbar, 50, on_approx_poly_epsilon_change) # Range 1-50
    cv2.createTrackbar("AR Tolerance (1-30->.01-.3)", "Live Video Feed", aspect_ratio_tolerance_trackbar, 30, on_aspect_ratio_tolerance_change)
    cv2.createTrackbar("Min Area % (1-50->.1-5%)", "Live Video Feed", min_area_percent_trackbar, 50, on_min_area_percent_change)

    print("\nPress 'q' to quit.")
    print("Adjust trackbars to tune detection parameters.")

    try:
        while True:
            frame = device.receive_scene_video_frame()
            if frame is None:
                # print("Warning: Failed to receive frame.") # Reduce console noise
                if cv2.waitKey(100) & 0xFF == ord('q'): 
                    break
                continue

            scene_img = frame.bgr_pixels
            display_img = scene_img.copy()

            # edged_output is still returned but will not be displayed in a separate window
            detected_corners, _ = detect_screen_corners_tuned(scene_img)

            if detected_corners is not None:
                # Draw the polygon
                cv2.polylines(display_img, [detected_corners.astype(np.int32)], True, (0, 255, 0), 2)
                # Draw and label corners
                corner_labels = ["TL", "TR", "BR", "BL"] # Assuming order_points gives TL, TR, BR, BL
                for i, point in enumerate(detected_corners):
                    pt_int = tuple(point.astype(int))
                    cv2.circle(display_img, pt_int, 5, (0, 0, 255), -1) # Red circles for corners
                    cv2.putText(display_img, corner_labels[i], (pt_int[0] + 10, pt_int[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # Blue labels
            else:
                # Display "No screen detected" message
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "No screen detected"
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = (display_img.shape[1] - text_size[0]) // 2
                text_y = (display_img.shape[0] + text_size[1]) // 2
                cv2.putText(display_img, text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
            cv2.imshow("Live Video Feed", display_img)
            # if edged_output is not None: # Removed this block

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing device and windows.")
        if device:
            device.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
