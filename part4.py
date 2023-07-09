import cv2
import numpy as np
from utils.image import (read_image, resize_image, show_image)
from part3 import find_edge_of_mask_img


def hough_line_func(image):
    height, width = image.shape[:2]
    masked_edges = find_edge_of_mask_img(image, height, width)
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi /
                            180, threshold=130, minLineLength=55, maxLineGap=30)

    # Draw the lines on the original image
    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            result = cv2.line(line_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    return result


if __name__ == '__main__':
    img1 = read_image('data/img1.jpg')
    resized_image = resize_image(img1)
    show_image('img1', resized_image)

    line_image = hough_line_func(resized_image)
    show_image('detected_line_img1', line_image)
