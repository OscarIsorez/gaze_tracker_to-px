import sys
import json

from pupil_labs.realtime_api.simple import discover_one_device
from pupil_labs.real_time_screen_gaze.gaze_mapper import GazeMapper

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import pyautogui

from ui import TagWindow
from dwell_detector import DwellDetector

pyautogui.FAILSAFE = False

class PupilPointerApp(QApplication):
    def __init__(self):
        super().__init__()

        self.setApplicationDisplayName('Pupil Pointer')
        self.mouseEnabled = False

        self.tagWindow = TagWindow()

        self.device = None
        self.dwellDetector = DwellDetector(.75, 75)
        self.smoothing = 0.8

        self.tagWindow.surfaceChanged.connect(self.onSurfaceChanged)

        self.tagWindow.dwellTimeChanged.connect(self.dwellDetector.setDuration)
        self.tagWindow.dwellRadiusChanged.connect(self.dwellDetector.setRange)
        self.tagWindow.mouseEnableChanged.connect(self.setMouseEnabled)
        self.tagWindow.smoothingChanged.connect(self.setSmoothing)

        self.last_timestamps = []

        self.pollTimer = QTimer()
        self.pollTimer.setInterval(1000 / 15)
        self.pollTimer.timeout.connect(self.poll)

        self.surface = None
        self.firstPoll = True

        self.mousePosition = None
        self.gazeMapper = None

    def onSurfaceChanged(self):
        self.updateSurface()

    def start(self):
        self.device = discover_one_device(max_search_duration_seconds=0.25)

        if self.device is None:
            QTimer.singleShot(1000, self.start)
            return

        calibration = self.device.get_calibration()
        self.gazeMapper = GazeMapper(calibration)

        self.tagWindow.setStatus(f'Connected to {self.device}. One moment...')

        self.updateSurface()
        self.pollTimer.start()
        self.firstPoll = True

    def updateSurface(self):
        if self.gazeMapper is None:
            return

        self.gazeMapper.clear_surfaces()
        self.surface = self.gazeMapper.add_surface(
            self.tagWindow.getMarkerVerts(),
            self.tagWindow.getSurfaceSize()
        )

    def setMouseEnabled(self, enabled):
        self.mouseEnabled = enabled

    def setSmoothing(self, value):
        self.smoothing = value

    def poll(self):
        frameAndGaze = self.device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1/15)
        data = self.device.receive_gaze_datum(timeout_seconds=1/15)
        if data:
            if len(self.last_timestamps) <2:
                self.last_timestamps.append(data.timestamp_unix_seconds)
            else:
                self.last_timestamps[0] = self.last_timestamps[1]
                self.last_timestamps[1] = data.timestamp_unix_seconds
            # if len(self.last_timestamps) == 2:
                # print(f"Timestamp 0: {self.last_timestamps[0]}")
                # print(f"Timestamp 1: {self.last_timestamps[1]}")
                # print(f"Timestamp diff: {self.last_timestamps[1] - self.last_timestamps[0]}")
                # print(f"Refresh rate (Hz) {1/(self.last_timestamps[1] - self.last_timestamps[0])}")
        estimate_offset = self.device.estimate_time_offset(
        )
        # print(f"Estimate offset  {estimate_offset}")

        
        if frameAndGaze is None:
            return

        else:
            self.tagWindow.setStatus(f'Streaming data from {self.device}')
            self.firstPoll = False

        frame, gaze = frameAndGaze
        result = self.gazeMapper.process_frame(frame, gaze)

        markerIds = [int(marker.uid.split(':')[-1]) for marker in result.markers]
        self.tagWindow.showMarkerFeedback(markerIds)

        if self.surface.uid in result.mapped_gaze:
            for surface_gaze in result.mapped_gaze[self.surface.uid]:
                if self.mousePosition is None:
                    self.mousePosition = [surface_gaze.x, surface_gaze.y]

                else:
                    self.mousePosition[0] = self.mousePosition[0] * self.smoothing + surface_gaze.x * (1.0 - self.smoothing)
                    self.mousePosition[1] = self.mousePosition[1] * self.smoothing + surface_gaze.y * (1.0 - self.smoothing)
                    print("surface gaze", surface_gaze.x, surface_gaze.y)

                mousePoint = self.tagWindow.updatePoint(*self.mousePosition)

                changed, dwell, dwellPosition = self.dwellDetector.addPoint(mousePoint.x(), mousePoint.y(), gaze.timestamp_unix_seconds)
                if changed and dwell:
                    self.tagWindow.setClicked(True)
                    if self.mouseEnabled:
                        pyautogui.click(x=dwellPosition[0], y=dwellPosition[1])
                else:
                    self.tagWindow.setClicked(False)

                if self.mouseEnabled:
                    QCursor().setPos(mousePoint)
            if len(result.mapped_gaze[self.surface.uid]) == 0:
                print("No gaze data")
    def exec(self):
        self.tagWindow.setStatus('Looking for a device...')
        self.tagWindow.showMaximized()
        QTimer.singleShot(1000, self.start)
        super().exec()
        if self.device is not None:
            self.device.close()

def run():
    app = PupilPointerApp()
    app.exec()
if __name__ == "__main__":
    run()