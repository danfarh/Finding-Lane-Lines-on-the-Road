import cv2
import numpy as np
from utils.image import (read_image, resize_image, show_image, grayscale_image)


def mask_img(image):
    height, width = image.shape[:2]
    polygons = np.array([[0, height], [width, height], [
                        width // 2, height // 1.5]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygons], (255, 0, 0))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def find_edge_of_mask_img(image, height, width):
    gray_image = grayscale_image(image)
    gaussian_blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(gaussian_blur_image, 50, 150)
    polygons = np.array([[0, height], [width, height], [
                        width // 2, height // 1.5]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [polygons], (255, 0, 0))
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image


if __name__ == '__main__':
    img1 = read_image('data/img1.jpg')
    resized_image = resize_image(img1)
    show_image('img1', resized_image)

    masked_img = mask_img(resized_image)
    show_image('masked_img1', masked_img)

    height, width = resized_image.shape[:2]
    edge_masked_img = find_edge_of_mask_img(resized_image, height, width)
    show_image('edge_masked_img1', edge_masked_img)
