# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, TryExcept, ops, plt_settings, threaded
from ultralytics.utils.checks import check_font, check_version, is_ascii
from ultralytics.utils.files import increment_path


class Colors:
    """
    Ultralytics color palette https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68", "FF6FDD", "FF444F", "CCED00", "00F344", "BD00FF",
            "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3", "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B")
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255], [153, 204, 255],
             [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
             [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51], [0, 255, 0], [0, 0, 255],
             [255, 0, 0], [255, 255, 255], ], dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if self.pil:
            self.im = im if input_is_pil else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]
        else:
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)
            self.sf = self.lw / 3

        self.colors = Colors()

        # Pose
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {(235, 219, 11), (243, 243, 243), (183, 223, 0), (221, 111, 255), (0, 237, 204),
                            (68, 243, 0), (255, 255, 0), (179, 255, 1), (11, 255, 162)}
        self.light_colors = {(255, 42, 4), (79, 68, 255), (255, 0, 189), (255, 180, 0), (186, 0, 221), (0, 192, 38),
                             (255, 36, 125), (104, 0, 123), (108, 27, 255), (47, 109, 252), (104, 31, 17)}

    def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
        return (104, 31, 17) if color in self.dark_colors else (
        255, 255, 255) if color in self.light_colors else txt_color

    def box_label(
            self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False, thickness=None
    ):
        if isinstance(box, torch.Tensor): box = box.tolist()
        if self.pil:
            self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = box[1] - h >= 0
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1, box[1] + 1 if outside else box[1] + h),
                    fill=color)
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:
            if rotated:
                box = cv2.boxPoints(box)
                box = np.int0(box)
                cv2.drawContours(self.im, [box], 0, color, thickness or self.lw, lineType=cv2.LINE_AA)
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=thickness or self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
                outside = p1[1] - h - 3 >= 0
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.sf, txt_color,
                            thickness=self.tf, lineType=cv2.LINE_AA)

    def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True, kpt_color=None):
        if self.pil:
            self.im = np.asarray(self.im).copy()
        nkpt = kpts.shape[1] if kpts.ndim > 1 else 0
        for i, k in enumerate(kpts):
            color_k = kpt_color or [int(x) for x in self.colors(i)]
            x_coord, y_coord = k[0], k[1]
            if x_coord > 0 and y_coord > 0:
                conf = k[2] if len(k) == 3 else 1
                if conf < 0.5:
                    continue

                # 使用 cv2 而不是 self.cv2
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

                label = f"{conf:.2f}"
                # 使用 cv2 而不是 self.cv2
                (w, h), _ = cv2.getTextSize(label, 0, fontScale=self.sf / 1.5, thickness=self.tf)

                outside = int(y_coord) - h - 3 < 0
                bg_p1 = (int(x_coord), int(y_coord) - h - 3 if not outside else int(y_coord) + 3)
                bg_p2 = (int(x_coord) + w, int(y_coord) if not outside else int(y_coord) + h + 3)
                # 使用 cv2 而不是 self.cv2
                cv2.rectangle(self.im, bg_p1, bg_p2, color_k, -1, cv2.LINE_AA)

                text_p1 = (int(x_coord) + 2, int(y_coord) - h if not outside else int(y_coord) + h + 2)
                # 使用 cv2 而不是 self.cv2
                cv2.putText(self.im, label, text_p1, 0, self.sf / 1.5, (255, 255, 255), thickness=self.tf,
                            lineType=cv2.LINE_AA)

        if kpt_line and nkpt > 0 and nkpt == 17:  # 增加 nkpt > 0 判断
            for j, (p1, p2) in enumerate(self.skeleton):
                if p1 < len(kpts) and p2 < len(kpts) and kpts[p1, 2] > 0.5 and kpts[p2, 2] > 0.5:
                    limb = self.limb_color[j]
                    # 使用 cv2 而不是 self.cv2
                    cv2.line(self.im, (int(kpts[p1, 0]), int(kpts[p1, 1])), (int(kpts[p2, 0]), int(kpts[p2, 1])),
                             limb, thickness=2, lineType=cv2.LINE_AA)
    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_style=False):
        """Adds text to an image using PIL or cv2."""
        if anchor == "bottom":  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if "\n" in text:
                lines = text.split("\n")
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                h += 3  # add pixels to pad text
                outside = xy[1] >= h  # label fits outside box
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title=None):
        """Show the annotated image."""
        im = Image.fromarray(np.asarray(self.im)[..., ::-1])  # Convert numpy array to PIL Image with RGB to BGR
        if IS_COLAB or IS_KAGGLE:  # can not use IS_JUPYTER as will run for all ipython environments
            try:
                display(im)  # noqa - display() function only available in ipython environments
            except ImportError as e:
                LOGGER.warning(f"Unable to display image in Jupyter notebooks: {e}")
        else:
            im.show(title=title)

    def save(self, filename="image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    @staticmethod
    def get_bbox_dimension(bbox=None):
        """
        Calculate the area of a bounding box.

        Args:
            bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.
            area (float): Area enclosed by the bounding box.
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw region line.

        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # Draw small circles at the corner points
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        Draw centroid point and track trails.

        Args:
            track (list): object tracking points for trails display
            color (tuple): tracks line color
            track_thickness (int): track line thickness value
        """
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0)):
        """
        Displays queue counts on an image centered at the points with customizable font size and colors.

        Args:
            label (str): Queue counts label.
            points (tuple): Region points for center point calculation to display text.
            region_color (tuple): RGB queue region color.
            txt_color (tuple): RGB text display color.
        """
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Draw text
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
        """
        Display the bounding boxes labels in parking management app.

        Args:
            im0 (ndarray): Inference image.
            text (str): Object/class name.
            txt_color (tuple): Display color for text foreground.
            bg_color (tuple): Display color for text background.
            x_center (float): The x position center point for bounding box.
            y_center (float): The y position center point for bounding box.
            margin (int): The gap between text and rectangle for better display.
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)

    def display_analytics(self, im0, text, txt_color, bg_color, margin):
        """
        Display the overall statistics for parking lots.

        Args:
            im0 (ndarray): Inference image.
            text (dict): Labels dictionary.
            txt_color (tuple): Display color for text foreground.
            bg_color (tuple): Display color for text background.
            margin (int): Gap between text and rectangle for better display.
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """
        Calculate the pose angle for object.

        Args:
            a (float) : The value of pose point a
            b (float): The value of pose point b
            c (float): The value o pose point c

        Returns:
            angle (degree): Degree value of angle between three points
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_specific_points(self, keypoints, indices=None, radius=2, conf_thres=0.25):
        """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list): Keypoints data to be plotted.
            indices (list, optional): Keypoint indices to be plotted. Defaults to [2, 5, 7].
            radius (int, optional): Keypoint radius. Defaults to 2.
            conf_thres (float, optional): Confidence threshold for keypoints. Defaults to 0.25.

        Returns:
            (numpy.ndarray): Image with drawn keypoints.

        Note:
            Keypoint format: [x, y] or [x, y, confidence].
            Modifies self.im in-place.
        """
        indices = indices or [2, 5, 7]
        points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thres]

        # Draw lines between consecutive points
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Draw circles for keypoints
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im

    def plot_workout_information(self, display_text, position, color=(104, 31, 17), txt_color=(255, 255, 255)):
        """
        Draw text with a background on the image.

        Args:
            display_text (str): The text to be displayed.
            position (tuple): Coordinates (x, y) on the image where the text will be placed.
            color (tuple, optional): Text background color
            txt_color (tuple, optional): Text foreground color
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, self.sf, self.tf)

        # Draw background rectangle
        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
            color,
            -1,
        )
        # Draw text
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)

        return text_height

    def plot_angle_and_count_and_stage(
        self, angle_text, count_text, stage_text, center_kpt, color=(104, 31, 17), txt_color=(255, 255, 255)
    ):
        """
        Plot the pose angle, count value, and step stage.

        Args:
            angle_text (str): Angle value for workout monitoring
            count_text (str): Counts value for workout monitoring
            stage_text (str): Stage decision for workout monitoring
            center_kpt (list): Centroid pose index for workout monitoring
            color (tuple, optional): Text background color
            txt_color (tuple, optional): Text foreground color
        """
        # Format text
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

        # Draw angle, count and stage text
        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
        )
        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
        )
        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
        )

    def seg_bbox(self, mask, mask_color=(255, 0, 255), label=None, txt_color=(255, 255, 255)):
        """
        Function for drawing segmented object in bounding box shape.

        Args:
            mask (np.ndarray): A 2D array of shape (N, 2) containing the contour points of the segmented object.
            mask_color (tuple): RGB color for the contour and label background.
            label (str, optional): Text label for the object. If None, no label is drawn.
            txt_color (tuple): RGB color for the label text.
        """
        if mask.size == 0:  # no masks to plot
            return

        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)
        if label:
            text_size, _ = cv2.getTextSize(label, 0, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
                (int(mask[0][0]) + text_size[0] // 2 + 10, int(mask[0][1] + 10)),
                mask_color,
                -1,
            )
            cv2.putText(
                self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1])), 0, self.sf, txt_color, self.tf
            )

    def sweep_annotator(self, line_x=0, line_y=0, label=None, color=(221, 0, 186), txt_color=(255, 255, 255)):
        """
        Function for drawing a sweep annotation line and an optional label.

        Args:
            line_x (int): The x-coordinate of the sweep line.
            line_y (int): The y-coordinate limit of the sweep line.
            label (str, optional): Text label to be drawn in center of sweep line. If None, no label is drawn.
            color (tuple): RGB color for the line and label background.
            txt_color (tuple): RGB color for the label text.
        """
        # Draw the sweep line
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

        # Draw label, if provided
        if label:
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),
                (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
                color,
                -1,
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf,
            )

    def plot_distance_and_line(
        self, pixels_distance, centroids, line_color=(104, 31, 17), centroid_color=(255, 0, 255)
    ):
        """
        Plot the distance and line on frame.

        Args:
            pixels_distance (float): Pixels distance between two bbox centroids.
            centroids (list): Bounding box centroids data.
            line_color (tuple, optional): Distance line color.
            centroid_color (tuple, optional): Bounding box centroid color.
        """
        # Get the text size
        text = f"Pixels Distance: {pixels_distance:.2f}"
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)

        # Define corners with 10-pixel margin and draw rectangle
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

        # Calculate the position for the text with a 10-pixel margin and draw text
        text_position = (25, 25 + text_height_m + 10)
        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )

        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255)):
        """
        Function for pinpoint human-vision eye mapping and plotting.

        Args:
            box (list): Bounding box coordinates
            center_point (tuple): center point for vision eye view
            color (tuple): object centroid and line color value
            pin_color (tuple): visioneye point color value
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)


@TryExcept()  # known issue https://github.com/ultralytics/yolov5/issues/5395
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """Plot training labels including class histograms and box statistics."""
    import pandas  # scope for faster 'import ultralytics'
    import seaborn  # scope for faster 'import ultralytics'

    # Filter matplotlib>=3.7.2 warning and Seaborn use_inf and is_categorical FutureWarnings
    warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # number of classes
    boxes = boxes[:1000000]  # limit to 1M boxes
    x = pandas.DataFrame(boxes, columns=["x", "y", "width", "height"])

    # Seaborn correlogram
    seaborn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # Matplotlib labels
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    seaborn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    seaborn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # Rectangles
    boxes[:, 0:2] = 0.5  # center
    boxes = ops.xywh2xyxy(boxes) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.

    Args:
        xyxy (torch.Tensor or list): A tensor or list representing the bounding box in xyxy format.
        im (numpy.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved. Defaults to 'im.jpg'.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box. Defaults to 1.02.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box. Defaults to 10.
        square (bool, optional): If True, the bounding box will be transformed into a square. Defaults to False.
        BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB. Defaults to False.
        save (bool, optional): If True, the cropped image will be saved to disk. Defaults to True.

    Returns:
        (numpy.ndarray): The cropped image.

    Example:
        ```python
        from ultralytics.utils.plotting import save_one_box

        xyxy = [50, 50, 150, 150]
        im = cv2.imread("image.jpg")
        cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
        ```
    """
    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop


@threaded
def plot_images(
    images: Union[torch.Tensor, np.ndarray],
    batch_idx: Union[torch.Tensor, np.ndarray],
    cls: Union[torch.Tensor, np.ndarray],
    bboxes: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.float32),
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    masks: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.uint8),
    kpts: Union[torch.Tensor, np.ndarray] = np.zeros((0, 51), dtype=np.float32),
    paths: Optional[List[str]] = None,
    fname: str = "images.jpg",
    names: Optional[Dict[int, str]] = None,
    on_plot: Optional[Callable] = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        images: Batch of images to plot. Shape: (batch_size, channels, height, width).
        batch_idx: Batch indices for each detection. Shape: (num_detections,).
        cls: Class labels for each detection. Shape: (num_detections,).
        bboxes: Bounding boxes for each detection. Shape: (num_detections, 4) or (num_detections, 5) for rotated boxes.
        confs: Confidence scores for each detection. Shape: (num_detections,).
        masks: Instance segmentation masks. Shape: (num_detections, height, width) or (1, height, width).
        kpts: Keypoints for each detection. Shape: (num_detections, 51).
        paths: List of file paths for each image in the batch.
        fname: Output filename for the plotted image grid.
        names: Dictionary mapping class indices to class names.
        on_plot: Optional callback function to be called after saving the plot.
        max_size: Maximum size of the output image grid.
        max_subplots: Maximum number of subplots in the image grid.
        save: Whether to save the plotted image grid to a file.
        conf_thres: Confidence threshold for displaying detections.

    Returns:
        np.ndarray: Plotted image grid as a numpy array if save is False, None otherwise.

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # scale to pixels
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color, rotated=is_obb)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)

            # Plot keypoints
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # if normalized with tolerance .01
                        kpts_[..., 0] *= w  # scale to pixels
                        kpts_[..., 1] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        try:
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # save
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None):
    """
    Plot training results from a results CSV file. The function supports various types of data including segmentation,
    pose estimation, and classification. Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to 'path/to/results.csv'.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided. Defaults to ''.
        segment (bool, optional): Flag to indicate if the data is for segmentation. Defaults to False.
        pose (bool, optional): Flag to indicate if the data is for pose estimation. Defaults to False.
        classify (bool, optional): Flag to indicate if the data is for classification. Defaults to False.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.
            Defaults to None.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results("path/to/results.csv", segment=True)
        ```
    """
    import pandas as pd  # scope for faster 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [2, 5, 3, 4]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 8, 9, 12, 13]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 9, 10, 13, 14]
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 9, 10, 11, 7, 8]
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in {8, 9, 10}:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.warning(f"WARNING: Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def plt_color_scatter(v, f, bins=20, cmap="viridis", alpha=0.8, edgecolors="none"):
    """
    Plots a scatter plot with points colored based on a 2D histogram.

    Args:
        v (array-like): Values for the x-axis.
        f (array-like): Values for the y-axis.
        bins (int, optional): Number of bins for the histogram. Defaults to 20.
        cmap (str, optional): Colormap for the scatter plot. Defaults to 'viridis'.
        alpha (float, optional): Alpha for the scatter plot. Defaults to 0.8.
        edgecolors (str, optional): Edge colors for the scatter plot. Defaults to 'none'.

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    # Calculate 2D histogram and corresponding colors
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [
        hist[
            min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1),
            min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1),
        ]
        for i in range(len(v))
    ]

    # Scatter plot
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


def plot_tune_results(csv_file="tune_results.csv"):
    """
    Plot the evolution results stored in a 'tune_results.csv' file. The function generates a scatter plot for each key
    in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on the plots.

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results. Defaults to 'tune_results.csv'.

    Examples:
        >>> plot_tune_results("path/to/tune_results.csv")
    """
    import pandas as pd  # scope for faster 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d

    def _save_one_file(file):
        """Save one matplotlib plot to 'file'."""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f"Saved {file}")

    # Scatter plots for each hyperparameter
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)
    num_metrics_columns = 1
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    x = data.values
    fitness = x[:, 0]  # fitness
    j = np.argmax(fitness)  # max fitness index
    n = math.ceil(len(keys) ** 0.5)  # columns and rows in plot
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + num_metrics_columns]
        mu = v[j]  # best single result
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # limit to 40 characters
        plt.tick_params(axis="both", labelsize=8)  # Set axis label size to 8
        if i % n != 0:
            plt.yticks([])
    _save_one_file(csv_file.with_name("tune_scatter_plots.png"))

    # Fitness vs iteration
    x = range(1, len(fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(x, fitness, marker="o", linestyle="none", label="fitness")
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ":", label="smoothed", linewidth=2)  # smoothing line
    plt.title("Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    _save_one_file(csv_file.with_name("tune_fitness.png"))


def output_to_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def output_to_rotated_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls, angle = o[:max_det].cpu().split((4, 1, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, box, angle, conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """
    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # all model heads
        if m in module_type:
            return
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save
