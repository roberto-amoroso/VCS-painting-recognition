# Testing MODE ON
from utils import print_next_step, step_generator, show_image, draw_lines
import cv2
import numpy as np
import os
import time


def mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level):
    """Groups pixels together by colour and location.

    This function takes an image and mean-shift parameters and returns a version
    of the image that has had mean shift segmentation performed on it.

    Mean shift segmentation clusters nearby pixels with similar pixel values
    sets them all to have the value of the local maxima of pixel value.

    Parameters
    ----------
    img: ndarray
        image to apply the Mean Shift Segmentation
    spatial_radius: int
        The spatial window radius
    color_radius: int
        The color window radius
    maximum_pyramid_level: int
        Maximum level of the pyramid for the segmentation

    Returns
    -------
    ndarray
        filtered “posterized” image with color gradients and fine-grain
        texture flattened

    Notes
    -------
    For details visit:
    - https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#pyrmeanshiftfiltering
    - https://docs.opencv.org/master/d7/d00/tutorial_meanshift.html

    """

    dst_img = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius, maximum_pyramid_level)
    return dst_img


def get_mask_largest_segment(img, color_difference=1, x_samples=8, skip_white=False):
    """Create a mask using the largest segment (this segment will be white).

    This is done by setting every pixel that is not the same color of the wall
    to have a value of 0 and every pixel has a value within a euclidean distance
    of `color_difference` to the wall's pixel value to have a value of 255.

    Parameters
    ----------
    img: ndarray
        image to apply masking
    color_difference: int
        euclidean distance between wall's pixel and the rest of the image
    x_samples : int
        numer of samples that will be tested orizontally in the image

    Returns
    -------
    ndarray
        Returns a version of the image where the wall is white and the rest of
        the image is black.
    """

    h, w, chn = img.shape
    color_difference = (color_difference,) * 3

    # in that way for smaller images the stride will be lower
    stride = int(w / x_samples)

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    wall_mask = mask
    largest_segment = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if mask[y + 1, x + 1] == 0 or skip_white:
                mask[:] = 0
                # Fills a connected component with the given color.
                # For details visit:
                # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill#floodfill
                rect = cv2.floodFill(
                    image=img.copy(),
                    mask=mask,
                    seedPoint=(x, y),
                    newVal=0,
                    loDiff=color_difference,
                    upDiff=color_difference,
                    flags=4 | (255 << 8),
                )

                # For details visit:
                # https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57

                # Next operation is not necessary if flag is equal to `4 | ( 255 << 8 )`
                # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                segment_size = mask.sum()
                if segment_size > largest_segment:
                    largest_segment = segment_size
                    wall_mask = mask[1:-1, 1:-1].copy()
                    # show_image('rect[2]', mask)
                    # cv2.waitKey(0)
    return wall_mask


def image_dilation(img, kernel_size):
    """Dilate the image.

    Dilation involves moving a kernel over the pixels of a binary image. When
    the kernel is centered on a pixel with a value of 0 and some of its pixels
    are on pixels with a value of 1, the centre pixel is given a value of 1.

    Parameters
    ----------
    img: ndarray
        the img to be dilated
    kernel_size: int
        the kernel size
    kernel_value: int
        the kernel value (the value of each kernel pixel)

    Returns
    -------
    ndarray
        Returns the dilated img
    """

    kernel = np.ones((kernel_size, kernel_size))
    dilated_img = cv2.dilate(img, kernel)
    return dilated_img


def image_erosion(img, kernel_size):
    """Erode the image.

    It's the opposite of dilation. Erosion involves moving a kernel over the
    pixels of a binary image. A pixel in the original image (either 1 or 0)
    will be considered 1 only if all the pixels under the kernel is 1,
    otherwise it is eroded (made to zero).

    Parameters
    ----------
    img: ndarray
        the img to be eroded
    kernel_size: int
        the kernel size
    kernel_value: int
        the kernel value (the value of each kernel pixel)

    Returns
    -------
    ndarray
        Returns the eroded image

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    """

    kernel = np.ones((kernel_size, kernel_size))
    eroded_img = cv2.erode(img, kernel)
    return eroded_img


def image_morphology_tranformation(img, operation, kernel_size):
    """Performs morphological transformations

    Performs morphological transformations of the image using an erosion
    and dilation.

    Parameters
    ----------
    img: ndarray
        the input image
    operation: int
        type of operation
    kernel_size: int
        the kernel size

    Returns
    -------
    ndarray
        the transformed image of the same size and type as source image

    """
    kernel = np.ones((kernel_size, kernel_size))
    transformed_img = cv2.morphologyEx(img, operation, kernel)
    return transformed_img


def image_blurring(img, ksize):
    """Blurs an image using the median filter.

    Parameters
    ----------
    img: ndarray
        the input image
    ksize
        aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...

    Returns
    -------
    ndarray
        the image blurred

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
    - https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
    """

    assert ksize > 1 and ksize % 2 != 0, "`ksize` should be odd and grater than 1"
    return cv2.medianBlur(img, ksize)


def invert_image(img):
    """Returns an inverted version of the image.

    The function calculates per-element bit-wise inversion of the input image.
    This means that black (=0) pixels become white (=255), and vice versa.

    In our case, we need to invert the wall mask for finding possible painting components.

    From OpenCV documentation (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html):
    "
        In OpenCV, finding contours is like finding white object from black
        background. So remember, object to be found should be white and background
        should be black.
    "

    Parameters
    ----------
    img: ndarray
        the image to invert

    Returns
    -------
    ndarray
        the inverted image
    """

    return cv2.bitwise_not(img)


def find_image_contours(img, mode, method):
    """Finds contours in a binary image.

    The function retrieves contours from a binary image (i.e. the wall mask we
    find before)

    Parameters
    ----------
    img: ndarray
        binary image in which to find the contours (i.e. the wall mask)
    mode: int
        Contour retrieval mode
    method: int
        Contour approximation method

    Returns
    -------
    contours: list
        Detected contours. Each contour is stored as a list of all the
        contours in the image. Each individual contour is a Numpy array
        of (x,y) coordinates of boundary points of the object.

    Notes
    -----
    Fot details visit:
    - https://docs.opencv.org/trunk/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
    - https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html

    """

    contours, _ = cv2.findContours(img, mode, method)
    return contours


def extract_candidate_painting_contours(img, contours, find_min_area_rect=False, width_min=100, height_min=100,
                                        area_percentage_min=0.6):
    """Find the contours that are candidated to be considered possible paintings.

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
    find_min_area_rect: bool
        determines whether to use `cv2.boundingRect(contour)` (False) or `cv2.minAreaRect(contour)` (True)
    width_min: int
        min width of the bounding rectangle a painting
    height_min: int
        min height of the bounding rectangle a painting
    area_percentage_min: float
        min area of the bounding rectangle that must be occupied by the contour area

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

    """
    img_copy = img.copy()
    h_img, w_img, _ = img.shape
    area_img = h_img * w_img
    area_rect_min = height_min * width_min
    candidate_painting_contours = []
    if contours:
        for contour in contours:
            if find_min_area_rect:
                rect = cv2.minAreaRect(contour)
                x, y = rect[0]
                w_rect, h_rect = rect[1]
                rotation_angle = rect[2]
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 2)
            else:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                # Draw rectangles on the image [MUST be a COPY of the image]
                img_copy = cv2.rectangle(img_copy, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

            area_rect = h_rect * w_rect
            if area_img > area_rect >= area_rect_min and cv2.contourArea(contour) >= (area_rect * area_percentage_min):
                candidate_painting_contours.append(contour)
    show_image('image_rectangles', img_copy)
    return candidate_painting_contours


def canny_edge_detection(img, threshold1, threshold2):
    """Finds edges in an image using the Canny algorithm.

    Parameters
    ----------
    img: ndarray
        the input image
    threshold1: int
        first threshold for the hysteresis procedure.
    threshold2:int
        second threshold for the hysteresis procedure.

    Returns
    -------
    ndarray
        returns an edge map that has the same size and type as `img`

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny
    - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
    """
    return cv2.Canny(img, threshold1, threshold2)


def find_hough_lines(img, probabilistic_mode=False, rho=1, theta=np.pi / 180, threshold=0, ratio_percentage=0.15):
    """Detect straight lines.

    Detect straight lines using the Standard or Probabilistic Hough
    Line Transform.

    Parameters
    ----------
    img: ndarray
        input image
    probabilistic_mode: bool
        determines whether to use the Standard (False) or the Probabilistic
        (True) Hough Transform
    rho: int
        distance resolution of the accumulator in pixels.
    theta: float
        angle resolution of the accumulator in radians.
    threshold: int
        accumulator threshold parameter. Only those rows that get enough
        votes ( >`threshold` ) are returned.
    ratio_percentage: float
        percentage of the image's larger side. The image is searched for
        lines who's length is at least a certain percentage of the image's
        larger side (default 15%).

    Returns
    -------
    ndarray
        Returns a NumPy.array of lines

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    - https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
    """

    h, w = img.shape
    if probabilistic_mode:
        img_ratio = np.max([h, w]) * ratio_percentage
        lines = cv2.HoughLinesP(img, rho, theta, threshold, img_ratio, img_ratio / 5.5)
    else:
        lines = cv2.HoughLines(img, rho, theta, threshold, None, 0, 0)

    draw_lines(img, lines, probabilistic_mode)

    return lines


def extend_image_lines(img, lines, probabilistic_mode, color_value=255):
    """Create a mask by extending the lines received.

    Create a mask of the same size of the image `img`, where the `lines` received
    have been drawn in order to cross the whole image. The color used to draw
    the lines is specified by `color_value`.

    Parameters
    ----------
    img: ndarray
        the input image
    lines: ndarray
        a NumPy.array of lines
    probabilistic_mode: bool
        determines whether to use the Standard (False) or the Probabilistic
        (True) Hough Transform
    color_value: tuple
        tuple (B,G,R) that specifies the color of the lines

    Returns
    -------
    ndarray
        Returns the mask with the lines drawn.

    Notes
    -----
    For details visit:
    - https://answers.opencv.org/question/2966/how-do-the-rho-and-theta-values-work-in-houghlines/#:~:text=rho%20is%20the%20distance%20from,are%20called%20rho%20and%20theta.
    """

    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    length = np.max((h, w))

    # TODO: improve the loop and remove redundancy
    for line in lines:
        line = line[0]
        if probabilistic_mode:
            theta = np.arctan2(line[1] - line[3], line[0] - line[2])

            x0 = line[0]
            y0 = line[1]

            a = np.cos(theta)
            b = np.sin(theta)

            pt1 = (int(x0 - length * a), int(y0 - length * b),)
            pt2 = (int(x0 + length * a), int(y0 + length * b),)
        else:
            rho = line[0]
            theta = line[1]

            a = np.cos(theta)
            b = np.sin(theta)

            # Read: https://answers.opencv.org/question/2966/how-do-the-rho-and-theta-values-work-in-houghlines/#:~:text=rho%20is%20the%20distance%20from,are%20called%20rho%20and%20theta.
            x0 = a * rho
            y0 = b * rho

            length = 1000

            pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
            pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))

        cv2.line(mask, pt1, pt2, color_value, 10, cv2.LINE_AA)

    return mask


def isolate_painting(img):
    """

    Parameters
    ----------
    img

    Returns
    -------

    """
    contours_mode = cv2.RETR_TREE
    contours_method = cv2.CHAIN_APPROX_NONE  # cv2.CHAIN_APPROX_SIMPLE
    painting_contours = find_image_contours(invert_image(img), contours_mode, contours_method)

    # TODO: [EVALUATE] If the mask does not look like we expect (like a sudoku puzzle)
    # then give up at this point :(
    # if len(painting_contours) < 9:
    #     return []

    largest_contour = painting_contours[0]
    for contour in painting_contours[1:]:
        largest_contour = contour if contour.shape[0] > largest_contour.shape[0] else largest_contour

    return largest_contour


def painting_db_lookup():
    """
    Function to lookup the DB for a specific painting
    Returns
    -------

    """
    pass


def find_painting_corners(img, mask):
    pass


def recognize_painting(img, mask, contours):
    """TODO: add descriotion and parameters needed

    Parameters
    ----------
    img
    mask
    contours

    Returns
    -------

    """
    for contour in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)

        sub_img = img[y:y + h_rect, x:x + w_rect]
        sub_mask = mask[y:y + h_rect, x:x + w_rect]

        cv2.imshow('image_sub_img', sub_img)
        cv2.imshow('image_sub_mask', sub_mask)

        # Step 9: Canny Edge detection to get the outline of the frame
        # ----------------------------
        print_next_step(generator, "Canny Edge detection:")
        start_time = time.time()
        # Credits: https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
        otsu_th, otsu_im = cv2.threshold(sub_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold1 = otsu_th * 0.5
        threshold2 = otsu_th
        edges = canny_edge_detection(sub_mask, threshold1, threshold2)
        exe_time_canny = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_canny))
        cv2.imshow('image_mask_canny', edges)

        # Step 10: Hough Lines to find vertical and horizontal edges of the paintings
        # ----------------------------
        print_next_step(generator, "Hough Lines:")
        start_time = time.time()
        probabilistic_mode = False
        rho = 1
        theta = np.pi / 180
        threshold = 50  # 60
        ratio_percentage = 0.10
        lines = find_hough_lines(
            img=edges,
            probabilistic_mode=probabilistic_mode,
            rho=rho,
            theta=theta,
            threshold=threshold,
            ratio_percentage=ratio_percentage
        )
        exe_time_hough = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_hough))

        # Step 11: Create mask from painting edges
        # ----------------------------
        print_next_step(generator, "Create mask from painting edges:")
        start_time = time.time()
        color_value = 255
        extended_lines_mask = extend_image_lines(sub_mask, lines, probabilistic_mode, color_value)
        exe_time_paint_mask = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_paint_mask))
        cv2.imshow('image_paint_mask', extended_lines_mask)

        # Step 12: Isolate Painting from mask
        # ----------------------------
        print_next_step(generator, "Isolate Painting from mask:")
        start_time = time.time()
        painting_contour = isolate_painting(extended_lines_mask)
        exe_time_painting_contour = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_painting_contour))
        # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
        img_contours = sub_img.copy()
        cv2.drawContours(img_contours, painting_contour, -1, (0, 255, 0), 3)
        show_image('painting_contours', img_contours)

        cv2.waitKey(0)


if __name__ == '__main__':
    photos_path = 'dataset/photos'
    videos_dir_name = '013'
    # filename = '20180529_112417_ok_0031.jpg'
    filename = '20180529_112417_ok_0026.jpg'

    generator = step_generator()
    total_time = 0

    img_path = os.path.join(photos_path, videos_dir_name, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(type(img))
    print(img.shape)
    show_image('image_original', img)

    # Step 1: Perform mean shift segmentation on the image
    # ----------------------------
    print_next_step(generator, "Mean Shift Segmentation:")
    start_time = time.time()
    spatial_radius = 7
    color_radius = 13
    maximum_pyramid_level = 1
    img_mss = mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level)
    exe_time_mean_shift_segmentation = time.time() - start_time
    total_time += exe_time_mean_shift_segmentation
    print("\ttime: {:.3f} s".format(exe_time_mean_shift_segmentation))
    show_image('image_mean_shift_segmentation', img_mss)

    # Step 2: Get a mask of just the wall in the gallery
    # ----------------------------
    print_next_step(generator, "Mask the Wall:")
    start_time = time.time()
    color_difference = 2
    x_samples = 8
    wall_mask = get_mask_largest_segment(img_mss, color_difference, x_samples)
    exe_time_mask_largest_segment = time.time() - start_time
    total_time += exe_time_mask_largest_segment
    print("\ttime: {:.3f} s".format(exe_time_mask_largest_segment))
    show_image('image_mask_largest_segment', wall_mask)

    # Step 3: Dilate and Erode the wall mask to remove noise
    # ----------------------------
    # IMPORTANT: We have not yet inverted the mask, therefore making
    # dilation at this stage is equivalent to erosion and vice versa
    # (we dilate the white pixels that are those of the wall).
    # ----------------------------
    print_next_step(generator, "Dilate and Erode:")
    kernel_size = 18
    start_time = time.time()
    dilated_wall_mask = image_dilation(wall_mask, kernel_size)
    exe_time_dilation = time.time() - start_time
    total_time += exe_time_dilation
    show_image('image_dilation', dilated_wall_mask)

    start_time = time.time()
    eroded_wall_mask = image_erosion(dilated_wall_mask, kernel_size)
    exe_time_erosion = time.time() - start_time
    total_time += exe_time_erosion
    print("\ttime: {:.3f} s".format(exe_time_dilation + exe_time_dilation))
    show_image('image_erosion', eroded_wall_mask)

    # Step 4: Invert the wall mask
    # ----------------------------
    print_next_step(generator, "Invert Wall Mask:")
    start_time = time.time()
    wall_mask_inverted = invert_image(eroded_wall_mask)
    exe_time_invertion = time.time() - start_time
    total_time += exe_time_invertion
    print("\ttime: {:.3f} s".format(exe_time_invertion))
    show_image('image_inversion', wall_mask_inverted)

    # ----------------------------
    # Connected Components Analysis:
    #   Perform connected components on the inverted wall mask to find the
    #   non-wall components
    # ----------------------------

    # Step 5: Find all contours
    # ----------------------------
    print_next_step(generator, "Find Contours:")
    start_time = time.time()
    contours_mode = cv2.RETR_TREE
    contours_method = cv2.CHAIN_APPROX_NONE  # cv2.CHAIN_APPROX_SIMPLE
    contours = find_image_contours(wall_mask_inverted, contours_mode, contours_method)
    exe_time_contours = time.time() - start_time
    total_time += exe_time_contours
    print("\ttime: {:.3f} s".format(exe_time_contours))
    # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    show_image('image_contours', img_contours)

    # Step 6: Refine list of components found
    # ----------------------------
    print_next_step(generator, "Refine Components found:")
    start_time = time.time()
    find_min_area_rect = True
    width_min = 100
    height_min = 100
    area_percentage_min = 0.5
    candidate_painting_contours = extract_candidate_painting_contours(
        img=img,
        contours=contours,
        find_min_area_rect=find_min_area_rect,
        width_min=width_min,
        height_min=height_min,
        area_percentage_min=area_percentage_min
    )
    exe_time_contours_refined = time.time() - start_time
    total_time += exe_time_contours_refined
    print("\ttime: {:.3f} s".format(exe_time_contours_refined))
    img_refined_contours = img.copy()
    cv2.drawContours(img_refined_contours, candidate_painting_contours, -1, (0, 255, 0), 3)
    show_image('image_refined_contours', img_refined_contours)

    # Step 7: Erode components to remove unwanted objects connected to the frame
    # ----------------------------
    print_next_step(generator, "Erode Components:")
    start_time = time.time()
    kernel_size = 40
    eroded_mask = image_erosion(wall_mask_inverted, kernel_size)
    exe_time_mask_erosion = time.time() - start_time
    total_time += exe_time_mask_erosion
    print("\ttime: {:.3f} s".format(exe_time_mask_erosion))
    show_image('image_mask_eroded', eroded_mask)

    # Step 8: Blur using Median Filter to smooth the lines of the frame
    # ----------------------------
    print_next_step(generator, "Blur with Median Filter:")
    start_time = time.time()
    blur_size = 31
    blurred_mask = image_blurring(eroded_mask, blur_size)
    exe_time_blurring = time.time() - start_time
    total_time += exe_time_blurring
    print("\ttime: {:.3f} s".format(exe_time_blurring))
    show_image('image_mask_blurred', blurred_mask)

    # ----------------------------
    # Recognize Painting:
    # for each frame contour, recognise a painting from it
    # ----------------------------
    start_time = time.time()
    recognize_painting(img, blurred_mask, candidate_painting_contours)
    exe_time_recognizing = time.time() - start_time
    total_time += exe_time_recognizing
    print("\n# Recognizing paintings total time: {:.3f} s".format(exe_time_recognizing))

    print()
    print("-" * 30)
    print("Total execution time: {:.3f} s".format(total_time))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
