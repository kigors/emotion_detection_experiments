import cv2
from pathlib import Path
import numpy as np
import time
from PIL import Image, ImageFont, ImageDraw 
from collections import deque
from emotion_classifier import EmotionClassifier


# opencv face detector wrapper
class FaceDetector:
    def __init__(self, folder='face_detection_data/') -> None:        
        """Load cv2 face detector."""
        if not Path(folder + 'opencv_face_detector.prototxt').exists() or \
            not Path(folder + 'opencv_face_detector.caffemodel').exists():
            raise ValueError('opencv detector files are missing')
        self.net = cv2.dnn.readNetFromCaffe(
                    folder + 'opencv_face_detector.prototxt', 
                    folder + 'opencv_face_detector.caffemodel'
                    )


    def get_boxes(self, image):
        """Run inference of face detector and return list of boxes with faces."""
        (h, w) = image.shape[:2]
        # preprocess image, add batch dim, rearrange dimentions (batch,channels,h,w)
        # size, mean and scale are determined by face detector 
        # (https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml)
        blob = cv2.dnn.blobFromImage(
            image=image, 
            scalefactor=1.0,
            size=(300, 300), 
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
        )
        
        # run inference of face detector
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # retrieve face boxes
        boxes = []
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int"))
        return boxes


# simple wrapper for opencv camera
class CameraWindow:
    def __init__(self, width=640, height=480, show_fps=True, selfie=False):
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            raise IOError('Failed to detect usb camera')
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width) # ширина кадра -- 640 пикселей
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # высота кадра -- 480 пикселей
        if show_fps:
            self.fps = deque(maxlen=15)
            self.prev_time = time.time()
        else:
            self.fps = None
            self.prev_time = None
        self.selfie = selfie


    def __del__(self):
        self.cam.release()


    def get_image(self):
        ret, frame =  self.cam.read()
        if ret and self.selfie:
            frame = cv2.flip(frame, 1)
        return ret, frame


    def show_image(self, image):
        if self.fps is not None:
            cur_time = time.time()
            self.fps.append(1 / (cur_time - self.prev_time))
            self.prev_time = cur_time
            cv2.putText(
                img=image, 
                text=f'FPS: {np.mean(self.fps):.0f}', 
                org=(15, 15), 
                fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=0.8, 
                color=(0, 0, 150), 
                thickness=1)
        
        cv2.imshow('Video - press q to quit', image)


class BoxDrawer:
    def __init__(self, font_height=15, font_ttf=None):
        """
        font_height in pixels - font size for a label in box's top bar.
        font_ttf - relative location of ttf file on disk
                   for a desired font (for example Google Fonts).
        """
        # text padding above and below in the bar
        self.padding = 2
        self.bar_height = font_height + 2 * self.padding
        
        # any ttf font you like, for example Google fonts.
        if font_ttf is None:
            font_ttf = 'fonts/Roboto_Condensed/RobotoCondensed-Regular.ttf'
            if not Path(font_ttf).exists():
                font_ttf = 'Arial.ttf' # one of Windows fontss
        try:
            self.title_font = ImageFont.truetype(font_ttf, font_height)
        except OSError:
            # better than nothing option
            self.title_font = ImageFont.load_default()
        
        # set background colors and font colors
        # you can change BoxDrawer.colors list. They are used in this order:
        #   one face -> color[0]
        #   two faces -> color[0] and color[1], etc.
        colors = [
            (74, 113, 173),
            (223, 132, 87),
            (80, 168, 108),
            (199, 78, 83),
            (130, 113, 176),
            (148, 120, 98),
            (220, 138, 193),
            (140, 140, 140),
            (205, 186, 121),
            (95, 181, 204)
        ]
        self.set_colors(colors)


    def set_colors(self, colors):
        """
        Set background colors and corresponding font colors based on relative luminosity.
        you can change BoxDrawer.colors list. They are used in this order:
          one face -> color[0]
          two faces -> color[0] and color[1], etc.
        colors - list of (R, G, B) where values are integers in [0, 255] range.
        """
        self.colors = colors
        dark_text_color = (14, 15, 16)
        light_text_color = (250, 250, 250)
        
        # simple choice of light or dark text color based on relative luminocity of background color
        lum = lambda r,g,b: (0.2126 * r + 0.7152 * g + 0.0722 * b)/255.
        rel_luminosity = np.array([lum(*i) for i in colors])
        text_colors = np.array([light_text_color] * len(colors))
        text_colors[rel_luminosity > 0.58] = dark_text_color

        self.text_colors = [tuple(i) for i in text_colors.tolist()]


    def draw_boxes_PIL(self, image, boxes, labels):
        """
        Draw rect and text with PIL lib. Much better quality than with cv2.
        image is in cv2 format (BGR channels).
        """

        my_image = Image.fromarray(image[..., ::-1])
        image_editable = ImageDraw.Draw(my_image)
        
        for i, (label, box) in enumerate(zip(labels, boxes)):
            frame_color = self.colors[i % len(self.colors)]
            text_color = self.text_colors[i % len(self.colors)]
            startX, startY, endX, endY = box
            
            if startY - self.bar_height  > 0:
                rectY = startY - self.bar_height 
            else:
                rectY = 0
                startY = self.bar_height
            textY = rectY + self.padding - 1

            # trim too long labels to fit bar width
            while image_editable.textlength(label, font=self.title_font) >= endX - startX - 5:
                label = label[:-3] + '..'

            image_editable.rectangle([startX, startY, endX, endY], fill=None, outline=frame_color, width=1)
            image_editable.rectangle([startX, startY, endX, rectY], fill=frame_color, outline=frame_color, width=1)
            image_editable.text((startX + 5, textY), label, text_color, font=self.title_font)

        return np.asarray(my_image)[..., ::-1].copy()


def box_scoring(box):
    """Assign a rank to the box based on its area and y-pos to avoid color flickering."""
    area = np.abs((box[0] - box[2]) * (box[1] - box[3]))
    return area + 0.001 * box[1]


def get_faces(frame, boxes):
    """Cut faces as per box coords, ignoring faces smaller than 10px and removing
    corresponding boxes from the list."""
    faces = []
    # when new face appear box size could be too small, need to be removed
    box_to_remove = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]

        # simple check of coords to avoid going out of index just in case
        x1 = x1 if x1 >= 0 else 0
        x2 = x2 if x2 < frame.shape[0] else frame.shape[0] - 1
        y1 = y1 if y1 >= 0 else 0
        y2 = y2 if y2 < frame.shape[1] else frame.shape[1] - 1

        # cut face and convert BGR -> RGB
        face = frame[x1:x2+1, y1:y2+1, ::-1]
        
        # do not count face and box if face is smaller than 10 px
        if min(face.shape[:-1]) > 10:                
            faces.append(frame[x1:x2+1, y1:y2+1, ::-1])
        else:
            box_to_remove.append(i)
    
    # remove too small boxes so indexing of faces and boxes will be the same
    for i in box_to_remove:
        boxes.pop(i)

    return faces


if __name__ == '__main__':

    print('Loading face detector...', end=' ')
    fd = FaceDetector(folder='face_detection_data/')
    
    print('Done.\nLoading tensorflow and classifier model...', end=' ')
    clf = EmotionClassifier('models/bit-m_ds2/config.ini')
    print('Done.')
    print(clf)
    
    print('Done.\nLoading camera...', end=' ')
    cam = CameraWindow(show_fps=True, selfie=True) # you can change width/height
    box_drawer = BoxDrawer(font_ttf='fonts/Roboto_Condensed/RobotoCondensed-Regular.ttf')
    print('Done.')

    emotion_frequency = 0.3
    previous_emo_time = time.time()

    while True:
        
        # Capture frame-by-frame
        ret, frame = cam.get_image()
        if ret == False:
            continue

        boxes = fd.get_boxes(frame)
        # sort boxes from large to small    
        boxes = sorted(boxes, key=box_scoring, reverse=True)

        faces = get_faces(frame, boxes)
        
        # limit emotion update rate to avoid label flickering too fast
        if time.time() - previous_emo_time >= emotion_frequency:
            if faces != []:
                labels, probabilities = clf.predict(faces)
            else:
                labels = []

        frame_new = box_drawer.draw_boxes_PIL(frame, boxes, labels)
        
        cam.show_image(frame_new)

        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('s'):
            # toggle frame flipping
            cam.selfie = not cam.selfie
        elif pressed_key == ord('q'):
            # quit program
            break
        
    # When everything is done, close window
    del cam # release camera
    cv2.destroyAllWindows()