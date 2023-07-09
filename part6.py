import cv2
from utils.video import (read_video, display_video, canny_func_video,
                         find_edge_of_mask_vid, better_hough_line_func_vid)


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


def find_edge_of_mask_video1(cap):
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            width = int(cap.get(3))
            height = int(cap.get(4))
            edges = find_edge_of_mask_vid(frame, height, width, 110, 200)
            cv2.imshow('Frame', edges)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def find_edge_of_mask_video2(cap):
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            width = int(cap.get(3))
            height = int(cap.get(4))
            edges = find_edge_of_mask_vid(frame, height, width, 50, 80)
            cv2.imshow('Frame', edges)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def better_hough_line_func_video1(cap):
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            width = int(cap.get(3))
            height = int(cap.get(4))
            lines = better_hough_line_func_vid(frame, height, width, 50, 150)
            cv2.imshow('Frame', lines)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def better_hough_line_func_video2(cap):
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            width = int(cap.get(3))
            height = int(cap.get(4))
            lines = better_hough_line_func_vid(frame, height, width, 50, 80)
            cv2.imshow('Frame', lines)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ################ VIDEO 1 ########################## 
    vid1 = read_video('data/vid1.mp4')
    display_video(vid1)
    canny_algorithm_vid1(vid1)
    find_edge_of_mask_video1(vid1)
    better_hough_line_func_video1(vid1)

    ################ VIDEO 2 ########################## 
    vid2 = read_video('data/vid2.mp4')
    display_video(vid2)
    canny_algorithm_vid2(vid2)
    find_edge_of_mask_video2(vid2)
    better_hough_line_func_video2(vid2)
