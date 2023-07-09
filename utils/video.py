import cv2
import numpy as np
from .image import grayscale_image
from part5 import calculate_avarage_slope_and_b, draw_line_on_image


def read_video(name):
    cap = cv2.VideoCapture(name)
    return cap


def display_video(cap):
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def canny_func_video(image, threshold1, threshold2):
    gray_image = grayscale_image(image)
    gaussian_blur_image = cv2.GaussianBlur(gray_image, (5, 5), 2)
    median_iamge = cv2.medianBlur(gaussian_blur_image, 5)
    edges = cv2.Canny(image=median_iamge,
                      threshold1=threshold1, threshold2=threshold2)
    return edges


def canny_algorithm_vid1(cap):
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            edges = canny_func_video(
                image=frame, threshold1=110, threshold2=200)
            cv2.imshow('Frame', edges)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def canny_algorithm_vid2(cap):
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            edges = canny_func_video(
                image=frame, threshold1=100, threshold2=120)
            cv2.imshow('Frame', edges)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def find_edge_of_mask_vid(image, height, width):
    edges = canny_func_video(image, 50, 150)
    polygons = np.array([[0, height], [width, height], [
                        width // 2, height // 1.5]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [polygons], (255, 0, 0))
    masked_vid = cv2.bitwise_and(edges, mask)
    return masked_vid


def better_hough_line_func_vid(image, image_height, image_width):
    masked_edges = find_edge_of_mask_vid(image, image_height, image_width)
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi /
                            180, threshold=120, minLineLength=55, maxLineGap=30)
    left_line, right_line = calculate_avarage_slope_and_b(image, lines)
    left_line_image = draw_line_on_image(image, left_line)
    line_image = draw_line_on_image(left_line_image, right_line)
    return line_image
