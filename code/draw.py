import cv2
import numpy as np
from scipy.spatial import distance as dist
import os
import pickle as pkl
import random


def step_generator():
    start = 1
    while True:
        yield start
        start += 1


def print_next_step(generator, title):
    # if title == "Hough Lines:":
    step = next(generator)
    print(f"\n# Step {step}: {title}")
    # print("-" * 30)
    # pass


def show_image(title, img, height=None, width=None, wait_key=True):
    """
    Create a window showing the given image with the given title.
    """
    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # if height is not None and width is not None:
    #     cv2.resizeWindow(title, width, height)
    # cv2.imshow(title, img)
    # if wait_key:
    #     cv2.waitKey(0)
    pass


def draw_people_bounding_box(img, people_bounding_boxes, scale_factor):
    """
    Draws the bounding box of people detected in the image.
    """
    colors = pkl.load(open("yolo/pallete", "rb"))
    for box in people_bounding_boxes:
        box = np.int32(np.array(box) * scale_factor)
        x, y, w, h = box

        label = "Person"
        color = random.choice(colors)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        font_scale = 1.5 * scale_factor
        line_thickness = 2
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, line_thickness)[0]
        c2 = x + t_size[0] + 3, y + t_size[1] + 4
        cv2.rectangle(img, (x, y), c2, color, -1)
        cv2.putText(img, label, (x, y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, font_scale, [225, 255, 255],
                    line_thickness)
    return img


def draw_lines(img, lines, probabilistic_mode=True):
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

    show_image("Detected Lines (in red) - {} Hough Line Transform".format(
        "Probabilistic" if probabilistic_mode else "Standard"), cdst, wait_key=False)


def draw_corners(img, corners):
    """
    Draws the corners on the given image.
    """
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)


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

    img_copy = img.copy()
    h = img_copy.shape[0]
    w = img_copy.shape[1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * scale_factor
    font_color = (0, 0, 0)
    line_thickness = 2

    for painting in paintings:
        corner_points = np.int32(painting.corners * scale_factor)

        if painting.title is not None:
            # Draw the title of the painting
            title = f"{painting.filename} - {painting.title}"

            # Find position of text above painting
            top = np.min(corner_points[:, 1])
            bottom = np.max(corner_points[:, 1])
            left = np.min(corner_points[:, 0])
            right = np.max(corner_points[:, 0])

            title_width, title_height = cv2.getTextSize(
                title,
                font,
                font_scale,
                line_thickness
            )[0]

            xb_title = int(left + (right - left) / 2 - title_width / 2)
            yb_title = int(top - title_height)

            # Check if the painting title is inside the video frame
            if yb_title - title_height < 0:
                yb_title = int(top + title_height + 15)

            cv2.rectangle(
                img_copy,
                (xb_title - 15, yb_title - title_height - 15),
                (xb_title + title_width + 15, yb_title + 15),
                (255, 255, 255),
                -1
            )

            bottom_left_corner_of_title = (xb_title, yb_title)

            # TODO: manage special character (like "ù") printed as "??"
            cv2.putText(img_copy,
                        title,
                        bottom_left_corner_of_title,
                        font,
                        font_scale,
                        font_color,
                        line_thickness)

        # Draw painting outline
        tl = tuple(corner_points[0])
        tr = tuple(corner_points[1])
        bl = tuple(corner_points[3])
        br = tuple(corner_points[2])
        cv2.line(img_copy, tl, tr, (0, 0, 255), 5)
        cv2.line(img_copy, tr, br, (0, 0, 255), 5)
        cv2.line(img_copy, br, bl, (0, 0, 255), 5)
        cv2.line(img_copy, bl, tl, (0, 0, 255), 5)

        show_image("partial_final_frame", img_copy, height=405, width=720)

        # cv2.waitKey(0)

    # Choose the room of the actual video frame by majority
    possible_rooms = [p.room for p in paintings if p.room is not None]
    if len(possible_rooms) > 0:
        major_room = max(possible_rooms, key=possible_rooms.count)
        room = f"Room: {major_room}"
    else:
        room = "Room: --"

    # Draw the room of the painting
    font_color = (0, 0, 0)
    room_width, room_height = cv2.getTextSize(
        room,
        font,
        font_scale,
        line_thickness
    )[0]
    xb_room = int(w / 2 - room_width / 2)
    yb_room = int(h - 20)

    bottom_left_corner_of_room = (xb_room, yb_room)

    cv2.rectangle(
        img_copy,
        (xb_room - 15, yb_room - room_height - 15),
        (xb_room + room_width + 15, h - 5),
        (255, 255, 255),
        -1
    )
    cv2.putText(img_copy,
                room,
                bottom_left_corner_of_room,
                font,
                font_scale,
                font_color,
                line_thickness)

    return img_copy
