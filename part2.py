import cv2
from utils.image import (read_image, resize_image, grayscale_image, show_image)


def canny_func(image):
    gray_image = grayscale_image(image)
    gaussian_blur_image = cv2.GaussianBlur(gray_image, (5, 5), 2)
    median_iamge = cv2.medianBlur(gaussian_blur_image, 5)
    edges = cv2.Canny(image=median_iamge, threshold1=50, threshold2=150)
    return edges


if __name__ == '__main__':
    img1 = read_image('data/img1.jpg')
    resized_image = resize_image(img1)
    show_image('img1', resized_image)

    canny_img = canny_func(resized_image)
    show_image('edge_img1', canny_img)
