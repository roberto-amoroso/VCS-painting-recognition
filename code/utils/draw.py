"""
Module containing functions to display, draw, show, and plot useful
information and data.
"""

import cv2
import numpy as np
import pickle as pkl
import random
import time
import matplotlib.pyplot as plt


def print_nicer(msg):
    """
    Print the message in a nicer way.
    """
    print("\n\n")
    print("-" * 50)
    print(f"# {msg}")


def print_time_info(start_time, msg=None, time_accumulator=None):
    """
    Print the time elapsed from `start_time` until now.
    """
    exe_time = time.time() - start_time
    if time_accumulator:
        time_accumulator += exe_time
    if msg:
        print(f"\n# {msg}")
    print("\tTime: {:.4f} s".format(exe_time))


def step_generator():
    """
    Generator returning an incremented counter at every call.
    """
    start = 1
    while True:
        yield start
        start += 1


def print_next_step_info(generator, title, same_line=False):
    """
    Print processing information at every call.
    """
    step = next(generator)
    if same_line:
        print(f"\tStep {step}: {title}\r", end='')
    else:
        print(f"\n\tStep {step}: {title}")
    # print("-" * 30)


def show_image_window(title, img, height=None, width=None, wait_key=True):
    """
    Create a window showing the given image with the given title.
    NO BLOCKING: all images are shown at the end of the script execution.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.axis('off')
    plt.title(title)
    plt.imshow(img_rgb)


def show_image_window_blocking(title, img, height=None, width=None, wait_key=True):
    """
    Create a window showing the given image with the given title.
    BLOCKING: each image is shown when it is created and a button
    must be pressed to continue the execution (mainly used for debugging)
    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if height is not None and width is not None:
        cv2.resizeWindow(title, width, height)
    cv2.imshow(title, img)
    if wait_key:
        cv2.waitKey(0)


def draw_people_bounding_box(img, people_bounding_boxes, scale_factor):
    """
    Draws the bounding box of people detected in the image.
    """
    for box in people_bounding_boxes:
        x, y, w, h = box

        label = "Person"
        color = (96, 128, 0)
        bbox_line_thickness = int(3 * scale_factor)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, bbox_line_thickness)

        font_scale = 1.5 * scale_factor
        line_thickness = int(2 * scale_factor)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, line_thickness)[0]
        c2 = x + t_size[0] + 3, y + t_size[1] + 4
        cv2.rectangle(img, (x, y), c2, color, -1)
        cv2.putText(img, label, (x, y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, font_scale, [225, 255, 255],
                    line_thickness)
    return img


def draw_lines(img, lines, probabilistic_mode=True):
    """
    Draw Hough lines on the received image. The lines could
    be obained with or without the Probabilistic version of the
    Hough algorithm.
    """
    # Draw the lines:
    # copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if probabilistic_mode:
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    else:
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                lenght = 20000
                pt1 = (int(x0 + lenght * (-b)), int(y0 + lenght * (a)))
                pt2 = (int(x0 - lenght * (-b)), int(y0 - lenght * (a)))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    return cdst


def draw_corners(img, corners):
    """
    Draws the corners on the given image.
    """
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)


def draw_room_info(img, people_room, scale_factor):
    """Draws information about the room where paintings and people are located.

        Parameters
        ----------
        img: ndarray
            the input image
        people_room: int or None
            number of the room where the paintings and people are located
        scale_factor: float
            scale factor for which the original image was scaled

        Returns
        -------
        ndarray
            a copy of the input image on which the information about the room
            were drawn.
    """

    h = img.shape[0]
    w = img.shape[1]

    if people_room != -1:
        if people_room is not None:
            room = f"Room: {people_room}"
        else:
            room = "Room: --"

        # Draw the room of the painting
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 2 * scale_factor
        line_thickness = int(2 * scale_factor)
        font_color = (0, 0, 0)
        room_width, room_height = cv2.getTextSize(
            room,
            font,
            font_scale,
            line_thickness
        )[0]
        # xb_room = int(w / 2 - room_width / 2)
        xb_room = int(20)
        yb_room = int(h - 20)

        bottom_left_corner_of_room = (xb_room, yb_room)

        cv2.rectangle(
            img,
            (xb_room - 15, yb_room - room_height - 15),
            (xb_room + room_width + 15, h - 5),
            (255, 255, 255),
            -1
        )
        cv2.putText(img,
                    room,
                    bottom_left_corner_of_room,
                    font,
                    font_scale,
                    font_color,
                    line_thickness)

    return img


def draw_paintings_info(img, paintings, scale_factor):
    """Draws all information about paintings found in the image.

    Parameters
    ----------
    img: ndarray
        the input image
    paintings: list
        list of painting found in the image
    scale_factor: float
        scale factor for which the original image was scaled

    Returns
    -------
    ndarray
        a copy of the input image on which all the information of
        the paintings found in it were drawn.

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html
    - https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    """

    h = img.shape[0]
    w = img.shape[1]

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1. * scale_factor
    font_color = (255, 255, 255)
    line_thickness = int(1 * scale_factor)

    for painting in paintings:
        corner_points = painting.corners

        if painting.title:
            # Color if painting is detected
            bbox_color = (153, 51, 0)
            # bbox_color = (5, 40, 255)
        else:
            # Color if painting is not detected
            bbox_color = (255, 102, 26)
            # bbox_color = (102, 133, 255)

        # Draw painting outline first, in order to not overlap painting info
        tl = tuple(corner_points[0])
        tr = tuple(corner_points[1])
        bl = tuple(corner_points[3])
        br = tuple(corner_points[2])
        bbox_line_thickness = int(3 * scale_factor)
        cv2.line(img, tl, tr, bbox_color, bbox_line_thickness)
        cv2.line(img, tr, br, bbox_color, bbox_line_thickness)
        cv2.line(img, br, bl, bbox_color, bbox_line_thickness)
        cv2.line(img, bl, tl, bbox_color, bbox_line_thickness)

        if painting.title is not None:
            # Draw the title of the painting
            # TODO remove filename info
            title = f"{painting.filename} - {painting.title}"

            # Find position of text above painting
            top = np.min(corner_points[:, 1])
            bottom = np.max(corner_points[:, 1])
            left = corner_points[0][0]
            # left = np.min(corner_points[:, 0])
            right = np.max(corner_points[:, 0])

            title_width, title_height = cv2.getTextSize(
                title,
                font,
                font_scale,
                line_thickness
            )[0]

            # xb_title = int(left + (right - left) / 2 - title_width / 2)
            xb_title = int(left)
            yb_title = int(top - title_height)

            padding = 5

            # Check if the painting title is inside the video frame
            if yb_title - title_height < 0:
                yb_title = int(top + title_height + padding)

            cv2.rectangle(
                img,
                (xb_title - padding, yb_title - title_height - padding),
                (xb_title + title_width + padding, yb_title + padding),
                bbox_color,
                -1
            )

            bottom_left_corner_of_title = (xb_title, yb_title)

            # TODO: manage special character (like "ù") printed as "??"
            cv2.putText(img,
                        title,
                        bottom_left_corner_of_title,
                        font,
                        font_scale,
                        font_color,
                        line_thickness)

        # show_image("partial_final_frame", img, height=405, width=720)

        # cv2.waitKey(0)
    return img
