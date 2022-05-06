from enum import Enum

import numpy as np
import cv2
from PIL import ImageGrab

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.tabbedpanel import TabbedPanel


class ImageMode(Enum):
    prepared = 1
    diff = 2
    thresh = 3
    full = 4


class Tools(BoxLayout):
    pass


def add_no_detection_rectangle(img_rgb, prepared_frame, previous_frame, rect, mult=1.0):
    s_f, e_f = rect
    s = (int(s_f[0] * mult), int(s_f[1] * mult))
    e = (int(e_f[0] * mult), int(e_f[1] * mult))
    mult_i = int(mult + 1)

    for i in range(s[0], e[0]):
        for j in range(s[1], e[1]):
            previous_frame[i, j] = 0
            prepared_frame[i, j] = 0
    for k in range(mult_i):
        for i in range(s[0], e[0]):
            if s[1] + k < len(img_rgb[0]):
                img_rgb[i, s[1] + k] = 100
            if e[1] + k < len(img_rgb[0]):
                img_rgb[i, e[1] + k] = 100
        for i in range(s[1], e[1]):
            if s[1] + k < len(img_rgb[0]):
                if s[0] + k < len(img_rgb[0]):
                    img_rgb[s[0] + k, i] = 100
                if e[0] + k < len(img_rgb[0]):
                    img_rgb[e[0] + k, i] = 100


class Detector(Image):
    def __init__(self, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.previous_frame = None
        self.thresh = 20
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.touch_down_pos = None
        self.rectangles = []

        self.source = "cam"
        self.fps = 30
        self.mode = ImageMode.full

        Clock.schedule_interval(self.update, 1.0 / self.fps)

    def clear_rects(self):
        self.rectangles.clear()

    def on_touch_down(self, touch):
        print("test")
        self.touch_down_pos = self.get_click_pos(touch)

    def on_touch_up(self, touch):
        if self.touch_down_pos is not None and self.get_click_pos(touch) is not None:
            self.rectangles.append((self.touch_down_pos, self.get_click_pos(touch)))

    def get_click_pos(self, touch):
        im_x = (self.size[0] - self.norm_image_size[0]) / 2.0 + self.x
        im_y = (self.size[1] - self.norm_image_size[1]) / 2.0 + self.y
        im_touch_x = touch.x - im_x
        im_touch_y = touch.y - im_y
        if im_touch_x < 0 or im_touch_x > self.norm_image_size[0]:
            return None
        elif im_touch_y < 0 or im_touch_y > self.norm_image_size[1]:
            return None
        else:
            return self.norm_image_size[1] - im_touch_y, im_touch_x

    def _get_frame(self, img_brg):
        img_rgb = img_brg
        if self.source == 'cam':
            img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

        prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

        if self.previous_frame is None:
            self.previous_frame = prepared_frame
            return

        for r in self.rectangles:
            add_no_detection_rectangle(img_rgb, prepared_frame, self.previous_frame, r,
                                       self.texture_size[0] / self.norm_image_size[0])

        diff_frame = cv2.absdiff(src1=self.previous_frame, src2=prepared_frame)
        self.previous_frame = prepared_frame

        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        thresh_frame = cv2.threshold(src=diff_frame, thresh=self.thresh, maxval=255, type=cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        if self.mode.value is ImageMode.prepared.value:
            return cv2.cvtColor(src=prepared_frame, code=cv2.COLOR_GRAY2RGB)
        elif self.mode.value is ImageMode.diff.value:
            return cv2.cvtColor(src=diff_frame, code=cv2.COLOR_GRAY2RGB)
        elif self.mode.value is ImageMode.thresh.value:
            return cv2.cvtColor(src=thresh_frame, code=cv2.COLOR_GRAY2RGB)
        else:
            return img_rgb

    def update(self, dt):
        if self.source == "cam":
            _, img_brg = self.cam.read()
        elif self.source == "screen":
            img_brg = np.array(ImageGrab.grab())
        else:
            print("wrong source")
            exit(0)

        frame = self._get_frame(img_brg)
        if frame is None:
            return
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture

    def change_source(self):
        self.clear_rects()
        if self.source == "cam":
            self.source = "screen"
        else:
            self.source = "cam"
        self.previous_frame = None


class ViewerWidget(BoxLayout):
    def change_source(self):
        for detector in self.detectors:
            self.ids[detector].change_source()

    def set_tresh(self, value):
        for detector in self.detectors:
            self.ids[detector].thresh = value


class DebugWidget(ViewerWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detectors = ['prepared_detector', 'thresh_detector', 'diff_detector', 'full_detector']


class DetectorWidget(ViewerWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detectors = ['detector']

class MainLayout(TabbedPanel):
    pass


class MotionDetectorApp(App):
    def build(self):
        return MainLayout()


if __name__ == '__main__':
    MotionDetectorApp().run()
