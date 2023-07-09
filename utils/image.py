import cv2


def read_image(name):
    image = cv2.imread(name)
    return image


def grayscale_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def resize_image(image):
    resized_image = cv2.resize(image, (600, 600))
    return resized_image


def show_image(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
