"""
Module containing functions to perform Painting Detection.
"""

from utils.draw import draw_corners, draw_lines
from tasks.image_processing import image_dilation, image_erosion, invert_image, find_image_contours, image_blurring, \
    canny_edge_detection, find_hough_lines, extend_image_lines, find_corners, \
    mean_shift_segmentation, find_largest_segment, create_segmented_image, image_morphology_tranformation
from utils.math_utils import order_points, calculate_polygon_area, translate_points
from models.painting import Painting

import cv2
import numpy as np
import time


def isolate_painting(mask):
    """Isolate painting from the mask of the image

    Parameters
    ----------
    mask: ndarray
        mask of the same size of the image. The painting should be white, the background black

    Returns
    -------
    list
        Largest contour found. Each contour is stored as a list of all the
        contours in the image. Each individual contour is a Numpy array
        of (x,y) coordinates of boundary points of the object.

    """
    contours_mode = cv2.RETR_TREE
    contours_method = cv2.CHAIN_APPROX_NONE  # cv2.CHAIN_APPROX_SIMPLE
    painting_contours, _ = find_image_contours(invert_image(mask), contours_mode, contours_method)

    # largest_contour = painting_contours[0]
    # for contour in painting_contours[1:]:
    #     largest_contour = contour if contour.shape[0] > largest_contour.shape[0] else largest_contour

    return max(painting_contours, key=cv2.contourArea)


def check_corners_area(img, contour, corners, min_percentage=0.8):
    """Checks the value of the area enclosed between the corners.

    Check if the corners of the image are 4 and cover at least the min_percentage of the contour
    area of the painting in the image. Otherwise this means there was a problem
    (e.g. image not squarish) and we return as corners the (tl, tr, br, bl)
    points of the image.

    Parameters
    ----------
    img: ndarray
        the input image
    contour: list
        contour of the painting in the image
    corners: ndarray
        a Numpy array of value (x, y)
    min_percentage: float
        min_percentage of the contour area of the painting in the image that must be
        covered by the area of the polygon identified by the corners.

    Returns
    -------
    ndarray
        checked corners as a Numpy array of value (x, y)
    """

    h = img.shape[0]
    w = img.shape[1]

    default_corners = True

    if corners is not None and corners.shape[0] == 4:
        corners = np.int32(order_points(corners[:, 0]))
        corners_area = calculate_polygon_area(corners)
        contour_area = cv2.contourArea(contour)

        if corners_area > contour_area * min_percentage:
            default_corners = False

    if default_corners:
        corners = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])

    return corners


def extract_candidate_painting_contours(img, contours, hierarchy, show_image, find_min_area_rect=False, width_min=100,
                                        height_min=100, area_percentage_min=0.6, remove_overlapping=False):
    """Find the contours that are candidated to be possible paintings.

    Returns the list of the contours that meet the following criteria:
    - not spanning the entire image
    - their bounding rectangles are bigger than `height_min` * `width_min`
    - take up more than `area_percentage_min` of their bounding rectangle


    If some paintings do not fall under these criteria, they may not be
    categorised as paintings.

    Parameters
    ----------
    img: ndarray
        the input image
    contours: list
        list of contours to check
    hierarchy: ndarray
        Representation of relationship between contours. OpenCV represents it as
        an array of four values :
            [Next, Previous, First_Child, Parent]
    show_image: function
        function used to show image of the intermediate results
    find_min_area_rect: bool
        determines whether to use `cv2.boundingRect(contour)` (False) or `cv2.minAreaRect(contour)` (True)
    width_min: int
        min width of the bounding rectangle a painting
    height_min: int
        min height of the bounding rectangle a painting
    area_percentage_min: float
        min area of the bounding rectangle that must be occupied by the contour area
    remove_overlapping: bool
        determine if remove or not overlapping contours

    Returns
    -------
    list
        Returns the list of the contours that meet the criteria described above

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
    - https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect#boundingrect
    - https://stackoverflow.com/questions/42453605/how-does-cv2-boundingrect-function-of-opencv-work
    - https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
    """
    # img_copy = img.copy()
    h_img, w_img, _ = img.shape
    area_img = h_img * w_img
    area_rect_min = height_min * width_min
    candidate_painting_contours = []

    if contours:
        hierarchy = [list(h) for h in hierarchy[0]]
        candidate_painting_hierarchy = []
        for h, contour in enumerate(contours):
            if find_min_area_rect:
                rect = cv2.minAreaRect(contour)
                x_center, y_center = rect[0]
                w_rect, h_rect = rect[1]
                rotation_angle = rect[2]
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 2)
            else:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                # Draw rectangles on the image [MUST be a COPY of the image]
                # img_copy = cv2.rectangle(img_copy, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

            # show_image('image_rectangles', img_copy, height=405, width=720)

            area_rect = h_rect * w_rect
            if area_img * 0.95 > area_rect >= area_rect_min and cv2.contourArea(contour) >= (
                    area_rect * area_percentage_min):
                candidate_painting_contours.append(contour)
                candidate_painting_hierarchy.append(list(hierarchy[h]))

        # Remove overlapping contours
        if remove_overlapping:
            for i, h in reversed(list(enumerate(candidate_painting_hierarchy))):
                # If at least one of the child of the current contour is a
                # candidate contour, then I remove the current contour
                # if h[2] != -1 and list(hierarchy[h[2]]) in candidate_painting_hierarchy:
                #     del candidate_painting_contours[i]
                if h[2] != -1:
                    # create a list of indices in candidate_painting_hierarchy
                    # of the childs of the current contour
                    contour_idx = hierarchy.index(h)
                    childs = [c for c in hierarchy if c[3] == contour_idx and c in candidate_painting_hierarchy]

                    if len(childs) > 0:
                        del candidate_painting_contours[i]

    # show_image_window_blocking('image_rectangles', img_copy, height=405, width=720)
    return candidate_painting_contours


def detect_paintings(img, generator, show_image, print_next_step, print_time, scale_factor=1.):
    """Detect Paintings in the image using OpenCV functions.

    Parameters
    ----------
    img: ndarray
        the input image
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

    Returns
    -------
    list
        a list containing one `Painting` object for each
        painting detected in the input image.

    """
    h_img, w_img, c_img = img.shape

    # Step 1: Perform mean shift segmentation on the image
    # ----------------------------
    print_next_step(generator, "Mean Shift Segmentation")
    start_time = time.time()
    spatial_radius = 7  # 8 # 8 # 5 #8 or 7
    color_radius = 14  # 15 # 40 #40 #35 or 15
    maximum_pyramid_level = 1  # 1
    img_mss = mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level)
    print_time(start_time)
    show_image('mean_shift_segmentation', img_mss, height=405, width=720)

    # Step 2: Get a mask of just the wall in the gallery
    # ----------------------------
    print_next_step(generator, "Mask the Wall")
    start_time = time.time()
    color_difference = 2  # 2 # 1
    x_samples = 8  # 8 or 16
    wall_mask = find_largest_segment(img_mss, color_difference, x_samples)
    print_time(start_time)
    show_image(f'mask_largest_segment', wall_mask, height=405, width=720)

    # Step 4: Invert the wall mask
    # ----------------------------
    print_next_step(generator, "Invert Wall Mask")
    start_time = time.time()
    wall_mask_inverted = invert_image(wall_mask)
    print_time(start_time)
    show_image('mask_inverted', wall_mask_inverted, height=405, width=720)

    # Step 3: Erode and Dilate the wall mask to remove noise
    # ----------------------------
    print_next_step(generator, "Erode and Dilate")
    kernel_size = 20  # 18 or 20

    start_time = time.time()
    eroded_wall_mask = image_erosion(wall_mask_inverted, kernel_size)
    print_time(start_time)
    show_image('mask_eroded', eroded_wall_mask, height=405, width=720)

    start_time = time.time()
    dilated_wall_mask = image_dilation(eroded_wall_mask, kernel_size)
    print_time(start_time)
    show_image('mask_dilated', dilated_wall_mask, height=405, width=720)

    wall_mask_inverted = dilated_wall_mask

    # ----------------------------
    # Connected Components Analysis:
    #   Perform connected components on the inverted wall mask to find the
    #   non-wall components
    # ----------------------------

    # Step 5: Find all contours
    # ----------------------------
    print_next_step(generator, "Find Contours")
    start_time = time.time()
    contours_mode = cv2.RETR_TREE
    contours_method = cv2.CHAIN_APPROX_NONE  # cv2.CHAIN_APPROX_SIMPLE
    contours_1, hierarchy_1 = find_image_contours(wall_mask_inverted, contours_mode, contours_method)
    # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
    # img_contours = img.copy()
    # cv2.drawContours(img_contours, contours_1, -1, (0, 255, 0), 3)
    # show_image('image_contours_1', img_contours, height=405, width=720)

    # Add a white border to manage cases when `find_largest_segment`
    # works the opposite way (wall black and painting white)
    thickness = 1
    wall_mask_inverted_2 = cv2.rectangle(wall_mask_inverted, (0, 0), (w_img - 1, h_img - 1), 255, thickness)
    # show_image("wall_mask_inverted_2", wall_mask_inverted_2, height=405, width=720)

    # Step 5: Find all contours
    # ----------------------------
    contours_mode = cv2.RETR_TREE
    contours_method = cv2.CHAIN_APPROX_NONE  # cv2.CHAIN_APPROX_SIMPLE
    contours_2, hierarchy_2 = find_image_contours(wall_mask_inverted_2, contours_mode, contours_method)
    print_time(start_time)
    # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
    # img_contours = img.copy()
    # cv2.drawContours(img_contours, contours_2, -1, (0, 255, 0), 3)
    # show_image('image_contours_2', img_contours, height=405, width=720)

    remove_overlapping = False
    error_in_wall_mask = False
    if len(contours_2) >= len(contours_1):
        contours = contours_2
        hierarchy = hierarchy_2
        remove_overlapping = True
        # Fix the wall mask considering the one before the inversion
        wall_mask_inverted = invert_image(dilated_wall_mask)
        error_in_wall_mask = True
        show_image('wall_mask_corrected', wall_mask_inverted, height=405, width=720)
    else:
        contours = contours_1
        hierarchy = hierarchy_1

    # # Print every contour step-by-step
    # img_contours = img.copy()
    # for contour in contours:
    #     cv2.drawContours(img_contours, [contour], 0, (0, 255, 0), 3)
    #     show_image('image_contours', img_contours, height=405, width=720)

    # Step 6: Refine list of components found
    # ----------------------------
    print_next_step(generator, "Refine Components found")
    start_time = time.time()
    find_min_area_rect = True  # True
    width_min = 150
    height_min = 150
    area_percentage_min = 0.6
    candidate_painting_contours = extract_candidate_painting_contours(
        img=img,
        contours=contours,
        hierarchy=hierarchy,
        show_image=show_image,
        find_min_area_rect=find_min_area_rect,
        width_min=width_min,
        height_min=height_min,
        area_percentage_min=area_percentage_min,
        remove_overlapping=remove_overlapping
    )
    print_time(start_time)
    img_refined_contours = img.copy()
    cv2.drawContours(img_refined_contours, candidate_painting_contours, -1, (0, 255, 0), 3)
    show_image('paintings_contours_refined', img_refined_contours, height=405, width=720)

    # Step SEGMENTATION: create a segmented image where only the candidate contours are white, in order to
    #                    remove unwanted object and make the following operation (erosion/dilation) faster
    # -----------------------
    print_next_step(generator, "Create Segmented Mask")
    start_time = time.time()
    segmented_img = create_segmented_image(wall_mask_inverted, candidate_painting_contours)
    # show_image('segmented_img', segmented_img, height=405, width=720)
    segmented_img = image_morphology_tranformation(segmented_img, cv2.MORPH_OPEN, 20)
    show_image('segmented_img', segmented_img, height=405, width=720)
    print_time(start_time)

    # I further refine the contours
    candidate_painting_contours, _ = find_image_contours(segmented_img, contours_mode, contours_method)

    # -----------------------
    # PADDING: add black padding to avoid unwanted "adherent" effects at the border when do erosion
    # -----------------------
    thickness = 1
    segmented_img = cv2.copyMakeBorder(segmented_img, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT,
                                       None, 0)

    # Step 7: Erode components to remove unwanted objects connected to the frame
    #         If there was an error in the wall mask (is inverted) then apply Dilation
    # ----------------------------
    print_next_step(generator, "Erode Components")
    start_time = time.time()
    kernel_size = 20  # 23 or 40
    if not error_in_wall_mask:
        cleaned_wall_mask = image_erosion(segmented_img, kernel_size)
    else:
        kernel_size = 30
        cleaned_wall_mask = image_dilation(segmented_img, kernel_size)
        cleaned_wall_mask = image_erosion(cleaned_wall_mask, kernel_size)
    print_time(start_time)
    # show_image('image_mask_cleaned', cleaned_wall_mask, height=405, width=720)

    # Remove padding
    cleaned_wall_mask = cleaned_wall_mask[thickness:-thickness, thickness:-thickness]

    # Step 8: Blur using Median Filter to smooth the lines of the frame
    # ----------------------------
    print_next_step(generator, "Blur with Median Filter")
    start_time = time.time()
    blur_size = 31  # 15
    blurred_mask = image_blurring(cleaned_wall_mask, blur_size)
    print_time(start_time)
    show_image('mask_eroded_and_blurred', blurred_mask, height=405, width=720)

    # ----------------------------
    # RECOGNIZE PAINTING:
    # for each frame contour, recognise a painting from it
    # ----------------------------

    paintings_detected = []
    for contour in candidate_painting_contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)

        sub_img = img[y:y + h_rect, x:x + w_rect]
        sub_mask = blurred_mask[y:y + h_rect, x:x + w_rect]

        show_image('sub_image', sub_img)
        show_image('sub_mask', sub_mask)

        # -----------------------
        # BORDER:
        # Add a black pixel of border in order to avoid problem
        # when you will try find edge of painting touching the border.
        # You can also use the `borderType=` parameter of `cv2.erode`
        # -----------------------
        thickness = 1
        pad_sub_mask = sub_mask.copy()
        cv2.rectangle(pad_sub_mask, (0, 0), (w_rect - 1, h_rect - 1), 0, thickness)

        # Step 9: Canny Edge detection to get the outline of the frame
        # ----------------------------
        print_next_step(generator, "Canny Edge detection")
        start_time = time.time()
        threshold1 = 70  # 50
        threshold2 = 140  # 100
        edges = canny_edge_detection(pad_sub_mask, threshold1, threshold2)
        print_time(start_time)
        show_image('mask_canny', edges)

        # Step 10: Hough Lines to find vertical and horizontal edges of the paintings
        # ----------------------------
        print_next_step(generator, "Hough Lines")
        start_time = time.time()
        probabilistic_mode = False
        rho = 1
        theta = np.pi / 180
        threshold = 50  # 50 or 30 or 40 or 0
        ratio_percentage = 0.10
        lines = find_hough_lines(
            img=edges,
            probabilistic_mode=probabilistic_mode,
            rho=rho,
            theta=theta,
            threshold=threshold,
            ratio_percentage=ratio_percentage
        )
        print_time(start_time)
        img_lines = draw_lines(edges, lines, probabilistic_mode)
        # show_image("Detected Lines (in red)", img_lines)

        if lines is None:
            # I can't find lines in special situation, e.g the painting is not squared (rounded, octagonal, ...)
            # In this case the corners are the tl, tr, br, bl point of `sub_img` and the contour is the original one
            corners = np.float32([
                [0, 0],
                [w_rect - 1, 0],
                [w_rect - 1, h_rect - 1],
                [0, h_rect - 1]
            ])
            painting_contour = contour
        else:
            # Step 11: Create mask from painting edges
            # ----------------------------
            print_next_step(generator, "Create mask from painting edges")
            start_time = time.time()
            color_value = 255
            extended_lines_mask = extend_image_lines(sub_mask, lines, probabilistic_mode, color_value)
            print_time(start_time)
            show_image('hough_lines_mask', extended_lines_mask)

            # Step 12: Isolate Painting from mask
            # ----------------------------
            print_next_step(generator, "Isolate Painting from mask")
            start_time = time.time()
            painting_contour = isolate_painting(extended_lines_mask)
            print_time(start_time)
            # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
            img_painting_contour = np.zeros((sub_img.shape[0], sub_img.shape[1]), dtype=np.uint8)
            cv2.drawContours(img_painting_contour, [painting_contour], 0, 255, cv2.FILLED)
            # If `cv2.drawContours` doesn't work, use `cv2.fillPoly`
            # cv2.fillPoly(img_painting_contour, pts=[painting_contour], color=255)
            show_image('painting_contours', img_painting_contour)

            # -----------------------
            # BORDER
            # -----------------------
            thickness = 1
            img_painting_contour = cv2.rectangle(img_painting_contour, (0, 0), (w_rect - 1, h_rect - 1), 0, thickness)

            # Step 13: Corner Detection of the painting
            # ----------------------------
            print_next_step(generator, "Corner Detection")
            start_time = time.time()
            max_number_corners = 4
            corner_quality = 0.001
            min_distance = 10  # 20
            corners = find_corners(
                img_painting_contour,
                max_number_corners=max_number_corners,
                corner_quality=corner_quality,
                min_distance=min_distance
            )
            # painting_corners = np.zeros((sub_img.shape[0], sub_img.shape[1]), dtype=np.uint8)
            # draw_corners(painting_corners, corners)
            # show_image('corners_TEST', painting_corners)

            # Checking corners to avoid problem (read function descr. for info)
            min_percentage = 0.70  # 0.7 or 0.75 or 0.6
            corners = check_corners_area(sub_img, contour, corners, min_percentage)
            print_time(start_time)

        # Draw painting corners
        painting_corners = np.zeros((sub_img.shape[0], sub_img.shape[1]), dtype=np.uint8)
        draw_corners(painting_corners, corners)
        show_image('painting_corners', painting_corners)

        # Create a new `Painting` object with all the information
        # about the painting detected
        detected_painting = Painting()
        detected_painting.bounding_box = np.int32(np.array([x, y, w_rect, h_rect]) * scale_factor)
        x, y, w_rect, h_rect = detected_painting.bounding_box
        detected_painting.frame_contour = np.int32(contour * scale_factor)
        detected_painting.points = translate_points(np.int32(painting_contour * scale_factor), [x, y])
        detected_painting.corners = translate_points(np.int32(corners * scale_factor), [x, y])
        paintings_detected.append(detected_painting)

        # cv2.waitKey(0)

    return paintings_detected
