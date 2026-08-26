"""Low-overhead, fixed-camera OpenCV preview for online mapping."""

import os
import sys
import threading

import cv2
import numpy as np


def compose_preview(rgb, metric_depth, valid, rendered_bgr, depth_min=0.2, depth_max=5.0):
    """Place camera RGB, metric depth, and Gaussian RGB side by side."""
    if depth_min < 0 or depth_max <= depth_min:
        raise ValueError('Preview depth range must satisfy 0 <= min < max')
    height, width = metric_depth.shape
    expected_rgb = (height, width, 3)
    if rgb.shape != expected_rgb or rendered_bgr.shape != expected_rgb:
        raise ValueError('Preview images must have matching dimensions')
    if valid.shape != metric_depth.shape:
        raise ValueError('Preview depth mask must match depth dimensions')

    normalized = np.zeros((height, width), dtype=np.uint8)
    normalized[valid] = np.clip((depth_max - metric_depth[valid]) * (255.0 / (depth_max - depth_min)), 0, 255).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_bgr[~valid] = 0

    canvas = np.empty((height, width * 3, 3), dtype=np.uint8)
    canvas[:, :width] = rgb[:, :, ::-1]
    canvas[:, width : 2 * width] = depth_bgr
    canvas[:, 2 * width :] = rendered_bgr
    labels = ('Camera RGB', 'Depth %.1f-%.1f m' % (depth_min, depth_max), 'Gaussian Render')
    for index, label in enumerate(labels):
        origin = (index * width + 10, 26)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


class PreviewVisualizer:
    """Own one asynchronous HighGUI window without blocking the mapper."""

    def __init__(self, depth_min=0.2, depth_max=5.0, window_name='GS-SLAM Monitor'):
        if depth_min < 0 or depth_max <= depth_min:
            raise ValueError('Preview depth range must satisfy 0 <= min < max')
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.window_name = window_name
        self.available = True
        self.opened = False
        self._condition = threading.Condition()
        self._pending = None
        self._thread = None
        self._closing = False
        self._exit_requested = False
        if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            self.available = False
            print('Preview disabled: no graphical display is available', flush=True)

    def show(self, rgb, metric_depth, valid, rendered_bgr):
        """Publish the newest frame and return False when the user requests exit.

        The single pending slot is deliberately overwritten when mapping is
        slower than the GUI.  A preview must show current state rather than
        replay an increasingly stale queue.
        """
        if not self.available:
            return True
        height, width = metric_depth.shape
        expected_rgb = (height, width, 3)
        if rgb.shape != expected_rgb or rendered_bgr.shape != expected_rgb:
            raise ValueError('Preview images must have matching dimensions')
        if valid.shape != metric_depth.shape:
            raise ValueError('Preview depth mask must match depth dimensions')
        with self._condition:
            if self._closing or self._exit_requested:
                return False
            self._pending = (rgb, metric_depth, valid, rendered_bgr)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='gs-slam-preview', daemon=True)
                self._thread.start()
            self._condition.notify()
        return True

    def _run(self):
        """Pump HighGUI events independently from long mapping iterations."""
        try:
            while True:
                with self._condition:
                    if self._pending is None and not self._closing:
                        self._condition.wait(timeout=0.02)
                    if self._closing:
                        break
                    pending = self._pending
                    self._pending = None

                if pending is not None:
                    canvas = compose_preview(*pending, self.depth_min, self.depth_max)
                    if not self.opened:
                        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                        self.opened = True
                    cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(15) & 0xFF
                if key in (27, ord('q')):
                    with self._condition:
                        self._exit_requested = True
                        self._pending = None
                    break
        except cv2.error as error:
            self.available = False
            print('Preview disabled: %s' % error, flush=True)
        finally:
            if self.opened:
                try:
                    cv2.destroyWindow(self.window_name)
                except cv2.error:
                    pass
                self.opened = False

    def close(self):
        with self._condition:
            self._closing = True
            self._pending = None
            self._condition.notify()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
