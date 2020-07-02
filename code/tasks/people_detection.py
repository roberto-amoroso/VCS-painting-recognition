"""
Module containing functions to perform People Detection.
"""

import time
import cv2
import numpy as np


def clean_people_bounding_box(img, paintings, people_bounding_boxes, max_percentage, scale_factor=1.):
    """
    Remove all the people bounding boxes which overlap more than
    `max_percentage` with one of the detected paintings.

    Parameters
    ----------
    img: ndarray
        the input image. Necessary to show the size of the mask to create.
    paintings: list
        list of painting detected
    people_bounding_boxes: list
        list of people bounding boxes
    max_percentage: float
        maximum percentage of overlap between the bounding box and one of
        the paintings
    scale_factor: float
        scale factor for which the original image was scaled

    Returns
    -------
    list
        the list of the valid people bounding boxes
    """

    h_img = img.shape[0]
    w_img = img.shape[1]
    clean_boxes = people_bounding_boxes[:]

    for painting in paintings:
        for box in people_bounding_boxes:
            x, y, w, h = box
            corner_points = np.int32(painting.corners / scale_factor)

            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            # show_image("test_1", mask)
            mask[y:y + h, x:x + w] = 255
            # show_image("test_2", mask)
            cv2.fillPoly(mask, pts=[corner_points], color=0)
            # show_image("test_3", mask)

            area_box = w * h
            white_pixels = np.sum(mask == 255)
            # If more than the max_percentage of the box is inside the
            # painting, than I will not consider this box as valid
            if area_box - white_pixels >= area_box * max_percentage:
                clean_boxes.remove(box)

        if len(clean_boxes) <= 0:
            break

    return clean_boxes


def detect_people(img, people_detector, paintings_detected, generator, show_image, print_next_step, print_time,
                  scale_factor=1, max_percentage=0.9):
    """Detect people in the image and predict a ROI around each person.

    Parameters
    ----------
    img: ndarray
        the input image
    people_detector: PeopleDetector
        `PeopleDetection` object using YOLOv3 to detect people in the image
    paintings_detected: list
        a list containing one `Painting` object for each
        painting detected in the input image.
    generator: generator
        generator function used to take track of the current step number
        and print useful information during processing.
    show_image: function
        function used to show image of the intermediate results
    print_next_step:function
        function used to print info about current processing step
    print_time: function
        function used to print info about execution time
    scale_factor: float
        scale factor for which the original image was scaled
    max_percentage: float
        maximum percentage of overlap between the bounding box and one of
        the paintings

    Returns
    -------
    list
        the list of the valid people bounding boxes
    """

    # Step YOLO: People Detection
    # ----------------------------
    print_next_step(generator, "YOLO People Detection")
    start_time = time.time()

    img_people_detected, people_in_frame, people_bounding_boxes = people_detector.run(img.copy())
    show_image('people_detection', img_people_detected, height=405, width=720)

    # Clean people bounding boxes only if I detected paintings
    if len(paintings_detected) > 0:
        # Step BOX: Clean bounding box to avoid overlap with painting
        people_bounding_boxes = clean_people_bounding_box(
            img,
            paintings_detected,
            people_bounding_boxes,
            max_percentage=max_percentage,
            scale_factor=scale_factor
        )

    people_bounding_boxes = np.int32(np.array(people_bounding_boxes) * scale_factor)

    print_time(start_time)

    return people_bounding_boxes
