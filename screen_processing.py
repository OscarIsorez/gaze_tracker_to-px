\
import cv2
import numpy as np

# --- Tunable Parameters (Global) ---
# Initial values from data_sender.py, meant to be updated by trackbars
canny_thr_1 = 5
canny_thr_2 = 25
blur_kernel_trackbar = 4  # Represents kernel size: 2*val + 1
approx_poly_epsilon_trackbar = 14 # Represents epsilon factor: val / 100.0
aspect_ratio_tolerance_trackbar = 26 # Represents tolerance factor: val / 100.0
min_area_percent_trackbar = 17 # Represents min area factor: val / 1000.0

# --- Callback functions for trackbars (to be called by UI in data_sender.py) ---
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
    """Sorts contour points to [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    # For top-right and bottom-left, we use the difference y-x
    # Top-right has smallest y-x (y is small, x is large)
    # Bottom-left has largest y-x (y is large, x is small)
    diff_yx = pts[:, 1] - pts[:, 0] 
    rect[1] = pts[np.argmin(diff_yx)] # Top-right
    rect[3] = pts[np.argmax(diff_yx)] # Bottom-left
    return rect

def detect_screen_corners(image):
    """
    Detects the four corners of a screen in an image using tunable parameters.
    Accesses global parameters defined in this module.
    """
    global canny_thr_1, canny_thr_2, blur_kernel_trackbar, \
           approx_poly_epsilon_trackbar, aspect_ratio_tolerance_trackbar, \
           min_area_percent_trackbar

    # Calculate current parameter values based on trackbar settings
    current_blur_kernel_size = 2 * blur_kernel_trackbar + 1
    if current_blur_kernel_size < 1: current_blur_kernel_size = 1 # Ensure kernel is at least 1x1

    current_approx_poly_epsilon = (approx_poly_epsilon_trackbar / 100.0)
    if current_approx_poly_epsilon < 0.01: current_approx_poly_epsilon = 0.01

    current_aspect_ratio_tolerance = aspect_ratio_tolerance_trackbar / 100.0
    if current_aspect_ratio_tolerance < 0.01: current_aspect_ratio_tolerance = 0.01
    
    current_min_area_factor = min_area_percent_trackbar / 1000.0 # e.g., 17 -> 0.017 (1.7%)
    if current_min_area_factor < 0.0001: current_min_area_factor = 0.0001

    target_aspect_ratio = 1920.0 / 1080.0 # Standard 16:9 screen

    # Image processing steps
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (current_blur_kernel_size, current_blur_kernel_size), 0)
    edged = cv2.Canny(blurred, canny_thr_1, canny_thr_2) # Uses global canny_thr_1, canny_thr_2
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None 

    contours = sorted(contours, key=cv2.contourArea, reverse=True) # Process largest contours first
    found_corners = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, current_approx_poly_epsilon * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2)
            ordered_corners = order_points(points.astype(np.float32)) # Use order_points from this module
            
            tl, tr, br, bl = ordered_corners
            
            # Calculate width and height of the detected quadrilateral
            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            avg_width = (width_top + width_bottom) / 2.0

            height_left = np.linalg.norm(bl - tl)
            height_right = np.linalg.norm(br - tr)
            avg_height = (height_left + height_right) / 2.0

            if avg_height == 0 or avg_width == 0: # Prevent division by zero
                continue
            
            aspect_ratio_detected = avg_width / avg_height
            
            # Check aspect ratio against target, within tolerance
            if abs(aspect_ratio_detected - target_aspect_ratio) <= current_aspect_ratio_tolerance * target_aspect_ratio:
                # Check minimum area to filter out small noise
                min_area_val = current_min_area_factor * image.shape[0] * image.shape[1]
                if cv2.contourArea(approx) < min_area_val:
                    continue # Contour is too small
                
                found_corners = ordered_corners.astype(np.float32)
                break # Found a suitable contour
    
    return found_corners
