# Testing MODE ON
from model.painting import Painting
import pandas as pd
from utils import print_next_step, step_generator, show_image, draw_lines, draw_corners, order_points, translate_points
import cv2
import numpy as np
import os
import time
from copy import copy
import matplotlib.pyplot as plt

generator = step_generator()


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    """Adjust automatically brightness and contrast of the image.

    Brightness and contrast is linear operator with parameter alpha and beta:
        g(x,y)= α * f(x,y)+ β

    It is recommended to visit the first link in the notes.

    Parameters
    ----------
    img: ndarray
        the input image

    Returns
    -------
    tuple
        new_img = adjusted image,
        alpha = alpha value calculated,
        beta = beta value calculated

    Notes
    -----
    For details visit.
    - https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    - https://answers.opencv.org/question/75510/how-to-make-auto-adjustmentsbrightness-and-contrast-for-image-android-opencv-image-correction/
    - https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#convertscaleabs
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    new_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_img, alpha, beta


def create_paintings_db(db_path, data_path):
    """Creates a list of all paintings in the DB and their info.

    Parameters
    ----------
    db_path: str
        path of the directory containing all painting files.
    data_path: str
        path of the '.csv' file containing all paintings info.

    Returns
    -------
    list
        list of `Painting` objects, describing all the paintings that populate
        the DB.
    """
    paintings_db = []
    df_painting_data = pd.read_csv(data_path)
    for subdir, dirs, files in os.walk(db_path):
        db_dir_name = subdir.replace('/', '\\').split('\\')[-1]

        print('Opened directory "{}"'.format(db_dir_name))

        for painting_file in files:
            image = cv2.imread(os.path.join(db_path, painting_file))
            painting_info = df_painting_data.loc[df_painting_data['Image'] == painting_file].iloc[0]
            title = painting_info['Title']
            author = painting_info['Author']
            room = painting_info['Room']
            painting = Painting(
                image,
                title,
                author,
                room,
                painting_file
            )
            paintings_db.append(painting)
    return paintings_db


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
    #                     show_image('rect[2]', mask, height=405, width=720)
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

    contours, hierarchy = cv2.findContours(img, mode, method)

    return contours, hierarchy


def extract_candidate_painting_contours(img, contours, hierarchy, find_min_area_rect=False, width_min=100,
                                        height_min=100, area_percentage_min=0.6, remove_overlapping=False):
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
    hierarchy: ndarray
        Representation of relationship between contours. OpenCV represents it as
        an array of four values :
            [Next, Previous, First_Child, Parent]
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
    img_copy = img.copy()
    h_img, w_img, _ = img.shape
    area_img = h_img * w_img
    area_rect_min = height_min * width_min
    candidate_painting_contours = []

    if contours:
        hierarchy = hierarchy[0]
        candidate_painting_hierarchy = []
        for h, contour in enumerate(contours):
            if find_min_area_rect:
                rect = cv2.minAreaRect(contour)
                x_center, y_center = rect[0]
                w_rect, h_rect = rect[1]
                rotation_angle = rect[2]
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 2)
            else:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                # Draw rectangles on the image [MUST be a COPY of the image]
                img_copy = cv2.rectangle(img_copy, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

            # show_image('image_rectangles', img_copy, height=405, width=720)

            area_rect = h_rect * w_rect
            if area_img * 0.95 > area_rect >= area_rect_min and cv2.contourArea(contour) >= (
                    area_rect * area_percentage_min):
                candidate_painting_contours.append(contour)
                candidate_painting_hierarchy.append(list(hierarchy[h]))

        # Remove overlapping contours
        if remove_overlapping:
            for i, h in reversed(list(enumerate(candidate_painting_hierarchy))):
                # If the child of the current contour is a candidate contour, then
                # I remove the current contour
                if h[2] != -1 and list(hierarchy[h[2]]) in candidate_painting_hierarchy:
                    del candidate_painting_contours[i]

    show_image('image_rectangles', img_copy, height=405, width=720)
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
        lines = cv2.HoughLinesP(img, rho, theta, threshold, img_ratio, img_ratio / 3.5)
    else:
        lines = cv2.HoughLines(img, rho, theta, threshold, None, 0, 0)

    # TODO: manage in a better way the draws of lines
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

            length = 2000

            pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
            pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))

        cv2.line(mask, pt1, pt2, color_value, 2, cv2.LINE_AA)  # cv2.LINE_AA

    return mask


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

    # TODO: [EVALUATE] If the mask does not look like we expect (like a sudoku puzzle)
    # then give up at this point :(
    # if len(painting_contours) < 9:
    #     return []

    # largest_contour = painting_contours[0]
    # for contour in painting_contours[1:]:
    #     largest_contour = contour if contour.shape[0] > largest_contour.shape[0] else largest_contour

    return max(painting_contours, key=cv2.contourArea)


def find_painting_corners(img, max_number_corners=4, corner_quality=0.001, min_distance=20):
    """Perform Shi-Tomasi Corner detection.

    Perform Shi-Tomasi Corner detection to return the corners found in the image.

    Parameters
    ----------
    img: ndarray
        the input image
    max_number_corners: int
        maximum number of corners to return
    corner_quality: float
        minimal accepted quality of image corners. The corners with the quality
        measure less than the product are rejected.
    min_distance: int
        minimum Euclidean distance between the returned corners

    Returns
    -------
    ndarray
        Returns a NumPy array of the most prominent corners in the image, in the form (x,y).

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    - https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html
    - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    """

    corners = cv2.goodFeaturesToTrack(
        img,
        max_number_corners,
        corner_quality,
        min_distance
    )
    return corners


def painting_rectification(src_img, corners, dst_img=None):
    """Executes Painting Rectification through an Affine Transformation.


    Returns a rectified version of the `src_img`. If The 'dst_img' is not None,
    the 'corners' of the `src_img` are translated to the corners of the
    'dst_img'. If The 'dst_img' is None the 'corners' of the `src_img` are
    used to calculate the aspect ratio of `src_img` and then rectify it using
    this information.

    Parameters
    -------
    src_img: ndarray
        source image to apply the transformation
    dst_img: ndarray
        destination image, the image used to transform the perspective of `src_img`
        After transform, `src_img` will have the same perspective as `dst_img`.
    corners: ndarray
        the NumPy array of the image corners

    Returns
    -------
    ndarray
        Returns an image that is like `src_img` but with the same perspective
        and shape as `dst_img`.

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective
    - https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography#findhomography
    - https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """

    # Source and destination points for the affine transformation
    src_points = np.float32(order_points(corners[:, 0]))
    (tl, tr, br, bl) = src_points

    if dst_img is not None:
        h_dst = dst_img.shape[0]
        w_dst = dst_img.shape[1]
    else:
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        w_dst = np.max((int(widthA), int(widthB))).clip(min=1)

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        h_dst = np.max((int(heightA), int(heightB))).clip(min=1)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst_points = np.float32([
        [0, 0],
        [w_dst - 1, 0],
        [w_dst - 1, h_dst - 1],
        [0, h_dst - 1]
    ])

    # Find perspective transformation
    retval, mask = cv2.findHomography(src_points, dst_points)

    # Apply perspective transformation to the image
    im_warped = cv2.warpPerspective(src_img, retval, (w_dst, h_dst), cv2.RANSAC)

    return im_warped


def match_features_orb(src_img, dst_img, max_distance=50):
    """Find the matches between two images.

    Find the matches between two images using ORB to find Keypoints.

    Parameters
    ----------
    src_img: ndarray
        source image
    dst_img: ndarray
        destination image
    max_distance: int
        minimum distance to consider a match valid
    Returns
    -------
    list
        Returns a list of all valid matched found.

    Notes
    -----
    For details visit
    - https://docs.opencv.org/3.4.0/d3/da1/classcv_1_1BFMatcher.html#ac6418c6f87e0e12a88979ea57980c020

    """
    orb = cv2.ORB_create()

    src_kp = orb.detect(src_img, None)
    src_kp, src_des = orb.compute(src_img, src_kp)
    show_image("src_kp", cv2.drawKeypoints(src_img, src_kp, None, color=(0, 255, 0), flags=0), wait_key=False)

    dst_kp = orb.detect(dst_img, None)
    dst_kp, dst_des = orb.compute(dst_img, dst_kp)
    show_image("dst_kp", cv2.drawKeypoints(dst_img, dst_kp, None, color=(0, 255, 0), flags=0), wait_key=False)

    # Find matches between the features in the source image and the destination
    # image (i.e. painting)
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(src_des, dst_des)
    matches = [m for m in matches if m.distance <= max_distance]

    # Show matches
    draw_params = dict(  # draw matches in green color
        singlePointColor=None,
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    matches_img = cv2.drawMatches(src_img, src_kp, dst_img, dst_kp, matches, None, **draw_params)
    show_image("matches_img", matches_img, wait_key=False)
    # cv2.waitKey(0)

    return matches


def histo_matching(src_img, dst_img):
    """Histogram Comparison of two images.

    Parameters
    ----------
    src_img: ndarray
        first image to compare
    dst_img: ndarray
        second image to compare

    Returns
    -------
    float
        Returns the value of the Histogram Comparison of two images using the
        Intersection method.

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    - https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html
    - https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
    - https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga6ca1876785483836f72a77ced8ea759a
    """

    # Convert image from BGR to HSV
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)

    # Set number of bins for hue and saturation
    hue_bins = 50
    sat_bins = 60
    hist_size = [hue_bins, sat_bins]

    # Set ranges for hue and saturation
    hue_range = [0, 180]
    sat_range = [0, 256]
    hist_range = hue_range + sat_range

    # Use the 0-th and 1-st channels of the histograms for the comparison
    channels = [0, 1]

    # Create histograms
    src_hist = cv2.calcHist(
        [src_img],
        channels,
        None,
        hist_size,
        hist_range,
        accumulate=False
    )
    cv2.normalize(src_hist, src_hist, 0, 1, cv2.NORM_MINMAX)

    dst_hist = cv2.calcHist(
        [dst_img],
        channels,
        None,
        hist_size,
        hist_range,
        accumulate=False
    )
    cv2.normalize(dst_hist, dst_hist, 0, 1, cv2.NORM_MINMAX)

    hist_comparison = cv2.compareHist(src_hist, dst_hist, cv2.HISTCMP_INTERSECT)
    return hist_comparison


def painting_db_lookup(img, corners, paintings_db, max_distance=40, match_db_image=False, histo_mode=False):
    """Lookup the DB for a specific painting using ORB or histogram matching.

    Parameters
    ----------
    img: ndarray
        the input image.
    corners: ndarray
        a NumPy array of the corners in `img`, in the form (x,y).
    paintings_db: list
        list of all `Painting` object that populate the DB.
    match_db_image: bool
        define if:
        - rectify `img` one time using aspect ratio (True)
        - rectify `img` for every painting in `paintings_db`
    histo_mode: bool
        indicate which method use for the matching:
        - True = Histogram Matching.
        - False = ORB matching.

    Returns
    -------
    ndarray
        Returns a NumPy array of tuple (id, num_matches), where id is the
        identification number of the current painting of the DB and
        num_matches is the amount of matches that `img` has with the current
        painting. The array is sorted in descending order of the number of
        matches.

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
    - https://numpy.org/doc/1.18/reference/generated/numpy.sort.html
    """

    dtype = [('painting_id', int), ('num_matches', float)]
    matches_rank = np.array([], dtype=dtype)

    if not match_db_image:
        # Step 15: Rectify Painting
        # ----------------------------
        print_next_step(generator, "Rectify Painting")
        start_time = time.time()
        rectified_img = painting_rectification(img, corners)
        exe_time_rectification = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_rectification))
        show_image('image_rectified', rectified_img)

    for id, painting_obj in enumerate(paintings_db):
        painting = painting_obj.image
        show_image('image_db', painting, wait_key=False)

        if match_db_image:
            # Step 15: Rectify Painting
            # ----------------------------
            print_next_step(generator, "Rectify Painting")
            start_time = time.time()
            rectified_img = painting_rectification(img, corners, painting)
            exe_time_rectification = time.time() - start_time
            print("\ttime: {:.3f} s".format(exe_time_rectification))
            show_image('image_rectified', rectified_img, wait_key=False)

        if not histo_mode:
            # Step 16: Match features using ORB
            # ----------------------------
            print_next_step(generator, "Match features using ORB")
            start_time = time.time()
            # max_distance = 40
            matches = match_features_orb(rectified_img, painting, max_distance)
            match_size = len(matches)
            exe_time_orb = time.time() - start_time
            print("\ttime: {:.3f} s".format(exe_time_orb))
        else:
            # Step 17: Match features using HISTOGRAMS
            # ----------------------------
            print_next_step(generator, "Match features using HISTOGRAMS")
            start_time = time.time()
            match_size = histo_matching(rectified_img, painting)
            exe_time_histo = time.time() - start_time
            print("\ttime: {:.3f} s".format(exe_time_histo))

        matches_rank = np.append(matches_rank, np.array([(id, match_size)], dtype=dtype))

    matches_rank = np.flip(np.sort(matches_rank, order='num_matches'))

    # If there is a best match, then return it
    if matches_rank[0][1] > 0:
        return matches_rank
    # Otherwise, return the id of painting having the most similar histogram
    else:
        return painting_db_lookup(img, corners, paintings_db, max_distance, match_db_image=match_db_image,
                                  histo_mode=True)


def recognize_painting(img, mask, contours, paintings_db):
    """Recognizes a painting from each frame contour.

    Parameters
    ----------
    img: ndarray
        the input image.
    mask: ndarray
        the mask of the image in which the wall is black and the possible
        paintings are white.
    contours: list
        the list of the contours of possible paintings found in the image.
    paintings_db: list
        list of all `Painting` object that populate the DB.

    Returns
    -------
    list
        Returns a list of all painting recognized in the input image `img`.

    """
    paintings_recognized = []
    for contour in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)

        sub_img = img[y:y + h_rect, x:x + w_rect]
        sub_mask = mask[y:y + h_rect, x:x + w_rect]

        # If the mask has more black pixels than white, I invert the mask
        if np.sum(sub_mask == 0) > np.sum(sub_mask == 255):
            sub_mask = invert_image(sub_mask)

        print_next_step(generator, "# Showing sub image")
        show_image('image_sub_img', sub_img)
        print_next_step(generator, "# Showing sub mask")
        show_image('image_sub_mask', sub_mask)

        # Step 0: Adjust automatically brightness and contrast of the image
        # ----------------------------
        print_next_step(generator, "Adjust brightness and contrast:")
        start_time = time.time()
        img_auto_adjusted, alpha, beta = automatic_brightness_and_contrast(sub_img)
        print(f"\talpha: {alpha}")
        print(f"\tbeta: {beta}")
        exe_time_auto_adjust = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_auto_adjust))
        show_image('auto_adjusted', img_auto_adjusted)

        sub_img = img_auto_adjusted

        # # Step 7: Erode components to remove unwanted objects connected to the frame
        # # ----------------------------
        # print_next_step(generator, "Erode Components:")
        # start_time = time.time()
        # kernel_size = 40
        # eroded_mask = image_erosion(sub_mask, kernel_size)
        # exe_time_mask_erosion = time.time() - start_time
                # print("\ttime: {:.3f} s".format(exe_time_mask_erosion))
        #         show_image()('image_mask_eroded', eroded_mask)
        #
        # # Step 8: Blur using Median Filter to smooth the lines of the frame
        # # ----------------------------
        # print_next_step(generator, "Blur with Median Filter:")
        # start_time = time.time()
        # blur_size = 31
        # blurred_mask = image_blurring(eroded_mask, blur_size)
        # exe_time_blurring = time.time() - start_time
                # print("\ttime: {:.3f} s".format(exe_time_blurring))
        #         show_image()('image_mask_blurred', blurred_mask)

        # -----------------------
        # BORDER:
        # Add a black pixel of border in order to avoid problem
        # when you will try find edge of painting touching the border.
        # You can also use the `borderType=` parameter of `cv2.erode`
        # -----------------------
        thickness = 1
        blurred_mask = cv2.rectangle(sub_mask, (0, 0), (w_rect - 1, h_rect - 1), 0, thickness)

        # Step 9: Canny Edge detection to get the outline of the frame
        # ----------------------------
        print_next_step(generator, "Canny Edge detection:")
        start_time = time.time()
        # Credits: https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
        otsu_th, otsu_im = cv2.threshold(blurred_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold1 = otsu_th * 0.5
        threshold2 = otsu_th
        edges = canny_edge_detection(blurred_mask, threshold1, threshold2)
        exe_time_canny = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_canny))
        show_image('image_mask_canny', edges)

        # Step 10: Hough Lines to find vertical and horizontal edges of the paintings
        # ----------------------------
        print_next_step(generator, "Hough Lines:")
        start_time = time.time()
        probabilistic_mode = False
        rho = 1
        theta = np.pi / 180
        threshold = 50  # 50 or 30 or 0
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

        if lines is None:
            print("# No lines found")
            continue

        # Step 11: Create mask from painting edges
        # ----------------------------
        print_next_step(generator, "Create mask from painting edges:")
        start_time = time.time()
        color_value = 255
        extended_lines_mask = extend_image_lines(sub_mask, lines, probabilistic_mode, color_value)
        exe_time_paint_mask = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_paint_mask))
        show_image('image_paint_mask', extended_lines_mask)

        # Step 12: Isolate Painting from mask
        # ----------------------------
        print_next_step(generator, "Isolate Painting from mask:")
        start_time = time.time()
        max_contour = isolate_painting(extended_lines_mask)
        exe_time_painting_contour = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_time_painting_contour))
        # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
        painting_contour = np.zeros((sub_img.shape[0], sub_img.shape[1]), dtype=np.uint8)
        cv2.drawContours(painting_contour, [max_contour], 0, 255, cv2.FILLED)
        # If `cv2.drawContours` doesn't work, use `cv2.fillPoly`
        # cv2.fillPoly(painting_contour, pts=[painting_contour], color=255)
        show_image('painting_contours', painting_contour)

        # -----------------------
        # BORDER
        # -----------------------
        thickness = 1
        painting_contour = cv2.rectangle(painting_contour, (0, 0), (w_rect - 1, h_rect - 1), 0, thickness)

        # Step 13: Corner Detection of the painting
        # ----------------------------
        print_next_step(generator, "Corner Detection")
        start_time = time.time()
        max_number_corners = 4
        corner_quality = 0.001
        min_distance = 20
        corners = find_painting_corners(
            painting_contour,
            max_number_corners=max_number_corners,
            corner_quality=corner_quality,
            min_distance=min_distance
        )
        exe_corner_detection = time.time() - start_time
        print("\ttime: {:.3f} s".format(exe_corner_detection))

        # If we found painting corners, then we execute DB lookup
        if corners is not None and corners.shape[0] == 4:
            painting_corners = np.zeros((sub_img.shape[0], sub_img.shape[1]), dtype=np.uint8)
            draw_corners(painting_corners, corners)
            show_image('painting_corners', painting_corners)

            # Step 14: Painting DB lookup
            # ----------------------------
            print_next_step(generator, "Painting DB lookup")
            start_time = time.time()
            max_distance = 40  # 40
            match_db_image = False  # False
            matches_rank = painting_db_lookup(
                sub_img,
                corners,
                paintings_db,
                max_distance=max_distance,
                match_db_image=match_db_image
            )
            painting_id = matches_rank[0][0]
            exe_db_lookup = time.time() - start_time
            print("\n# Painting DB lookup total time: {:.3f} s".format(exe_db_lookup))
            if painting_id is not None and painting_id != -1:
                recognized_painting = paintings_db[painting_id]

                i = 1
                # Manage case when I find duplicated painting in the save video frame
                while recognized_painting in paintings_recognized:
                    # At each iteration I select the next painting that had the highest number of matches
                    painting_id = matches_rank[i][0]
                    i += 1

                    if i == matches_rank.size:
                        recognized_painting = copy(paintings_db[painting_id])
                        break
                    else:
                        recognized_painting = paintings_db[painting_id]

                show_image("prediction", recognized_painting.image)
                recognized_painting.frame_contour = contour
                recognized_painting.points = translate_points(max_contour, [x, y])
                recognized_painting.corners = translate_points(corners, [x, y])
                paintings_recognized.append(recognized_painting)
        else:
            print("# Error in corners found")

        # cv2.waitKey(0)
    return paintings_recognized


def draw_paintings_info(img, paintings):
    """

    Parameters
    ----------
    img
    paintings

    Returns
    -------

    Notes
    -----
    For details visit:
    - https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html
    - https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    """

    img_copy = img.copy()
    h = img_copy.shape[0]
    w = img_copy.shape[1]

    # Choose the room of the actual video frame by majority
    possible_rooms = [p.room for p in paintings]
    major_room = max(possible_rooms, key=possible_rooms.count)
    room = f"Room: {major_room}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)
    line_thickness = 2

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

    i = 15
    for painting in paintings:
        corner_points = np.int32(order_points(painting.corners[:, 0]))

        # Find position of text above painting
        top = np.min(corner_points[:, 1])
        bottom = np.max(corner_points[:, 1])
        left = np.min(corner_points[:, 0])
        right = np.max(corner_points[:, 0])

        # Draw the title of the painting
        title = f"{painting.filename} - {painting.title}"
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
        # bottom_left_corner_of_title = (i, i)
        # i = i + 15

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

    return img_copy


if __name__ == '__main__':
    photos_path = 'dataset/photos'
    recognized_painting_path = 'dataset/recognized_paintings'
    videos_dir_name = '014'  # '013' or '009' or '014'
    filename = None
    # filename = '20180529_112417_ok_0031.jpg'
    # filename = '20180529_112417_ok_0026.jpg'
    # filename = 'IMG_2653_0002.jpg'
    # filename = 'IMG_2657_0006.jpg'
    # filename = 'IMG_2659_0012.jpg'  # CRITIC
    # filename = 'IMG_2659_0006.jpg'
    # filename = 'VID_20180529_113001_0000.jpg'  # LOTS painting not recognized
    # filename = "VID_20180529_112517_0004.jpg"
    # filename = "VID_20180529_112517_0005.jpg"
    # filename = "VID_20180529_112553_0002.jpg"  # Wall inverted
    # filename = "VID_20180529_112739_0004.jpg"  # Wall inverted
    filename = "VID_20180529_112627_0000.jpg"  # Wall correct
    # filename = "VID_20180529_112517_0002.jpg"  # strange case
    # filename = "VID_20180529_112553_0005.jpg"
    # filename = "IMG_2646_0004.jpg"
    # filename = "IMG_2646_0003.jpg" # overlapping contours
    # filename = "IMG_2646_0006.jpg" # overlapping contours

    painting_db_path = "./paintings_db"
    painting_data_path = "./data/data.csv"

    paintings_db = create_paintings_db(painting_db_path, painting_data_path)
    total_time = 0

    dst_dir = os.path.join(recognized_painting_path, videos_dir_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print('Created the directory "{}"'.format(dst_dir))
    else:
        print('The directory "{}" already exists'.format(dst_dir))

    for subdir, dirs, files in os.walk(os.path.join(photos_path, videos_dir_name)):
        for photo in files:
            if filename is not None:
                img_path = os.path.join(photos_path, videos_dir_name, filename)
            else:
                img_path = os.path.join(subdir, photo)

            print("\n#", "-" * 30)
            print(f"# Processing imag: {img_path}")
            print("#", "-" * 30, "\n")

            img_original = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h_img, w_img, c_img = img_original.shape
            print(f"Image shape: {img_original.shape}")
            show_image('image_original', img_original, height=405, width=720)

            # # Step 0: Adjust automatically brightness and contrast of the image
            # # ----------------------------
            # print_next_step(generator, "Adjust brightness and contrast:")
            # start_time = time.time()
            # img_auto_adjusted, alpha, beta = automatic_brightness_and_contrast(img_original)
            # print(f"\talpha: {alpha}")
            # print(f"\tbeta: {beta}")
            # exe_time_auto_adjust = time.time() - start_time
            # total_time += exe_time_auto_adjust
                        # print("\ttime: {:.3f} s".format(exe_time_auto_adjust))
            # show_image('image_auto_adjusted', img_auto_adjusted, height=405, width=720)

            # TODO: if auto-adjust will be in the `recognize_painting` function, than remove the following assigment
            # and change the name og `img_original` in `img`
            img = img_original

            # Step 1: Perform mean shift segmentation on the image
            # ----------------------------
            print_next_step(generator, "Mean Shift Segmentation:")
            start_time = time.time()
            spatial_radius = 7  # 7
            color_radius = 13  # 13
            maximum_pyramid_level = 1  # 1
            img_mss = mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level)
            exe_time_mean_shift_segmentation = time.time() - start_time
            total_time += exe_time_mean_shift_segmentation
            print("\ttime: {:.3f} s".format(exe_time_mean_shift_segmentation))
            show_image('image_mean_shift_segmentation', img_mss, height=405, width=720)

            # Step 2: Get a mask of just the wall in the gallery
            # ----------------------------
            print_next_step(generator, "Mask the Wall:")
            start_time = time.time()
            color_difference = 2  # 2
            x_samples = 8  # 8 or 16
            wall_mask = get_mask_largest_segment(img_mss, color_difference, x_samples)
            exe_time_mask_largest_segment = time.time() - start_time
            total_time += exe_time_mask_largest_segment
            print("\ttime: {:.3f} s".format(exe_time_mask_largest_segment))
            show_image('image_mask_largest_segment', wall_mask, height=405, width=720)

            # Step 3: Dilate and Erode the wall mask to remove noise
            # ----------------------------
            # IMPORTANT: We have not yet inverted the mask, therefore making
            # dilation at this stage is equivalent to erosion and vice versa
            # (we dilate the white pixels that are those of the wall).
            # ----------------------------
            print_next_step(generator, "Dilate and Erode:")
            kernel_size = 20  # 18 or 20
            start_time = time.time()
            dilated_wall_mask = image_dilation(wall_mask, kernel_size)
            exe_time_dilation = time.time() - start_time
            total_time += exe_time_dilation
            show_image('image_dilation', dilated_wall_mask, height=405, width=720)

            start_time = time.time()
            eroded_wall_mask = image_erosion(dilated_wall_mask, kernel_size)
            exe_time_erosion = time.time() - start_time
            total_time += exe_time_erosion
            print("\ttime: {:.3f} s".format(exe_time_dilation + exe_time_dilation))
            show_image('image_erosion', eroded_wall_mask, height=405, width=720)

            # Step 4: Invert the wall mask
            # ----------------------------
            print_next_step(generator, "Invert Wall Mask:")
            start_time = time.time()
            wall_mask_inverted = invert_image(eroded_wall_mask)
            exe_time_invertion = time.time() - start_time
            total_time += exe_time_invertion
            print("\ttime: {:.3f} s".format(exe_time_invertion))
            show_image('image_inversion', wall_mask_inverted, height=405, width=720)

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
            contours_1, hierarchy_1 = find_image_contours(wall_mask_inverted, contours_mode, contours_method)
            exe_time_contours = time.time() - start_time
            total_time += exe_time_contours
            print("\ttime: {:.3f} s".format(exe_time_contours))
            # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
            img_contours = img.copy()
            cv2.drawContours(img_contours, contours_1, -1, (0, 255, 0), 3)
            show_image('image_contours_1', img_contours, height=405, width=720)

            # TODO: test if keep or remove
            # Add a white border to manage cases when `get_mask_largest_segment`
            # works the opposite way (wall black and painting white)

            thickness = 1
            wall_mask_inverted_2 = cv2.rectangle(wall_mask_inverted, (0, 0), (w_img - 1, h_img - 1), 255, thickness)
            show_image("wall_mask_inverted_2", wall_mask_inverted_2, height=405, width=720)

            # Step 5: Find all contours
            # ----------------------------
            print_next_step(generator, "Find Contours:")
            start_time = time.time()
            contours_mode = cv2.RETR_TREE
            contours_method = cv2.CHAIN_APPROX_NONE  # cv2.CHAIN_APPROX_SIMPLE
            contours_2, hierarchy_2 = find_image_contours(wall_mask_inverted_2, contours_mode, contours_method)
            exe_time_contours = time.time() - start_time
            total_time += exe_time_contours
            print("\ttime: {:.3f} s".format(exe_time_contours))
            # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
            img_contours = img.copy()
            cv2.drawContours(img_contours, contours_2, -1, (0, 255, 0), 3)
            show_image('image_contours_2', img_contours, height=405, width=720)

            remove_overlapping = False
            if len(contours_2) >= len(contours_1):
                contours = contours_2
                hierarchy = hierarchy_2
                remove_overlapping = True
            else:
                contours = contours_1
                hierarchy = hierarchy_1

            # # Print every contour step-by-step
            # for contour in contours:
            #     cv2.drawContours(img_contours, [contour], 0, (0, 255, 0), 3)
            #     show_image('image_contours', img_contours, height=405, width=720)

            # Draw the contours on the image (https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html)
            img_contours = img.copy()
            cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
            show_image('image_contours', img_contours, height=405, width=720)

            # Step 6: Refine list of components found
            # ----------------------------
            print_next_step(generator, "Refine Components found:")
            start_time = time.time()
            find_min_area_rect = True  # True
            width_min = 150
            height_min = 150
            area_percentage_min = 0.6
            candidate_painting_contours = extract_candidate_painting_contours(
                img=img,
                contours=contours,
                hierarchy=hierarchy,
                find_min_area_rect=find_min_area_rect,
                width_min=width_min,
                height_min=height_min,
                area_percentage_min=area_percentage_min,
                remove_overlapping=remove_overlapping
            )
            exe_time_contours_refined = time.time() - start_time
            total_time += exe_time_contours_refined
            print("\ttime: {:.3f} s".format(exe_time_contours_refined))
            img_refined_contours = img.copy()
            cv2.drawContours(img_refined_contours, candidate_painting_contours, -1, (0, 255, 0), 3)
            show_image('image_refined_contours', img_refined_contours, height=405, width=720)

            # Step 7: Erode components to remove unwanted objects connected to the frame
            # ----------------------------
            print_next_step(generator, "Erode Components:")
            start_time = time.time()
            kernel_size = 40
            eroded_mask = image_erosion(wall_mask_inverted, kernel_size)
            exe_time_mask_erosion = time.time() - start_time
            total_time += exe_time_mask_erosion
            print("\ttime: {:.3f} s".format(exe_time_mask_erosion))
            show_image('image_mask_eroded', eroded_mask, height=405, width=720)

            # Step 8: Blur using Median Filter to smooth the lines of the frame
            # ----------------------------
            print_next_step(generator, "Blur with Median Filter:")
            start_time = time.time()
            blur_size = 31
            blurred_mask = image_blurring(eroded_mask, blur_size)
            exe_time_blurring = time.time() - start_time
            total_time += exe_time_blurring
            print("\ttime: {:.3f} s".format(exe_time_blurring))
            show_image('image_mask_blurred', blurred_mask, height=405, width=720)

            # ----------------------------
            # Recognize Painting:
            # for each frame contour, recognise a painting from it
            # ----------------------------
            start_time = time.time()
            paintings_recognized = recognize_painting(img, blurred_mask, candidate_painting_contours, paintings_db)
            exe_time_recognizing = time.time() - start_time
            total_time += exe_time_recognizing
            print("\n# Recognizing paintings total time: {:.3f} s".format(exe_time_recognizing))

            if len(paintings_recognized) > 0:
                # Step 18: Draw information about Paintings found
                # ----------------------------
                print_next_step(generator, "Draw paintings information:")
                start_time = time.time()
                final_frame = draw_paintings_info(img_original, paintings_recognized)
                exe_time_draw_info = time.time() - start_time
                total_time += exe_time_draw_info
                print("\ttime: {:.3f} s".format(exe_time_draw_info))
                # show_image('final_frame', final_frame, height=405, width=720)
                cv2.imshow('final_frame', final_frame)
            else:
                final_frame = img_original

            if filename is not None:
                break

            save_path = os.path.join(dst_dir, photo)
            cv2.imwrite(save_path, final_frame)

    print()
    print("-" * 30)
    print("Total execution time: {:.3f} s".format(total_time))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
#
