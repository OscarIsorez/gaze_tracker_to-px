\
import cv2
import numpy as np

class UIManager:
    def __init__(self, window_name="Live Video Feed + Gaze Sender"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        print(f"UIManager initialized for window: {self.window_name}")

    def setup_trackbars(self, initial_params, callback_map):
        """
        Creates OpenCV trackbars.
        initial_params: dict of {trackbar_name: initial_value}
        callback_map: dict of {trackbar_name: callback_function}
        """
        for name, initial_value in initial_params.items():
            max_val = 255 # Default max value for Canny thresholds
            if "Blur Kernel" in name: max_val = 4
            elif "Approx Poly Eps" in name: max_val = 50
            elif "AR Tolerance" in name: max_val = 30
            elif "Min Area %" in name: max_val = 50
            
            cv2.createTrackbar(name, self.window_name, initial_value, max_val, callback_map[name])
        print("Trackbars created.")

    def display_image(self, image):
        """Displays the image in the OpenCV window."""
        cv2.imshow(self.window_name, image)

    def draw_detection_info(self, display_img, detected_corners, homography_valid):
        """Draws screen detection status, corners, and labels on the image."""
        if detected_corners is not None:
            cv2.polylines(display_img, [detected_corners.astype(np.int32)], True, (0, 255, 0), 2)
            corner_labels = ["TL", "TR", "BR", "BL"]
            for i, point in enumerate(detected_corners):
                pt_int = tuple(point.astype(int))
                cv2.circle(display_img, pt_int, 5, (0, 0, 255), -1)
                cv2.putText(display_img, corner_labels[i], (pt_int[0] + 10, pt_int[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if homography_valid:
                cv2.putText(display_img, "Screen Detected. Sending Gaze.", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_img, "Screen Detected. Homography Failed.", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # orange
        else:
            cv2.putText(display_img, "No screen detected. Adjust parameters.", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return display_img

    def get_keypress(self, delay_ms=30):
        """Waits for a key press for a specified delay."""
        return cv2.waitKey(delay_ms) & 0xFF

    def destroy_windows(self):
        """Closes all OpenCV windows."""
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")

    def show_instructions(self):
        print("\nPress 'q' to quit.")
        print("Adjust trackbars to tune detection parameters.")
        print("Gaze data will be sent only when the screen is detected and homography is computed.")

