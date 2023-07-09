import cv2
import numpy as np
from utils.image import (read_image, resize_image, show_image)
from part3 import find_edge_of_mask_img


def make_coordinates(image, line_parameters):
    try:
        slope, b = line_parameters
    except TypeError:
        slope, b = 0.0001, 0

    y1 = image.shape[0]
    y2 = int(y1*(3.57 / 5))
    x1 = int((y1 - b)/slope)
    x2 = int((y2 - b)/slope)
    
    return np.array([x1, y1, x2, y2])


def calculate_avarage_slope_and_b(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            b = parameters[1]
            if slope < 0:
                left_fit.append((slope, b))
            else:
                right_fit.append((slope, b))

    left_fit_avarage = np.average(left_fit, axis=0)
    right_fit_avarage = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_avarage)
    right_line = make_coordinates(image, right_fit_avarage)
    return np.array([left_line, right_line])


def draw_line_on_image(image, coordinates):
    final_line_image = np.copy(image)
    result = cv2.line(final_line_image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]),
                      color=(0, 0, 255), thickness=2)
    return result


def better_hough_line_func(image, image_height, image_width):
    masked_edges = find_edge_of_mask_img(image, image_height, image_width)
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi /
                            180, threshold=120, minLineLength=55, maxLineGap=30)
    left_line, right_line = calculate_avarage_slope_and_b(image, lines)
    left_line_image = draw_line_on_image(image, left_line)
    line_image = draw_line_on_image(left_line_image, right_line)
    return line_image


if __name__ == '__main__':
    img1 = read_image('data/img1.jpg')
    resized_image = resize_image(img1)
    show_image('img1', resized_image)

    height, width = resized_image.shape[:2]
    detected_final_line_img1 = better_hough_line_func(
        resized_image, height, width)
    show_image('detected_line_img1', detected_final_line_img1)
