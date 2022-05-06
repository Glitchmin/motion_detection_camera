import numpy as np
import cv2
from PIL import ImageGrab


def add_no_detection_rectangle(img_rgb, prepared_frame, previous_frame, s_x, e_x, s_y, e_y):
    for i in range(s_x, e_x):
        for j in range(s_y, e_y):
            previous_frame[i, j] = 0
            prepared_frame[i, j] = 0
    for i in range(s_x, e_x):
        img_rgb[i, s_y] = 100
        img_rgb[i, e_y] = 100
    for i in range(s_y, e_y):
        img_rgb[s_x, i] = 100
        img_rgb[e_x, i] = 100


def motion_detector():
    frame_count = 0
    previous_frame = None

    while True:
        frame_count += 1

        img_brg = np.array(ImageGrab.grab())
        img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

        if (frame_count % 2) == 0:
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            if previous_frame is None:
                previous_frame = prepared_frame
                continue

            add_no_detection_rectangle(img_rgb, prepared_frame, previous_frame, 0, 200, 0, 200)

            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)

            thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

            cv2.imshow('Motion detector', img_rgb)

        if cv2.waitKey(30) == 27:
            break


motion_detector()
