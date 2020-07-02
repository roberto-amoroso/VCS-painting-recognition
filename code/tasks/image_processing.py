"""
This module contains generic image processing functions that can also be used
for applications other than Painting Detection.
"""

import cv2
import numpy as np


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
    ksize: int
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


#
def canny_edge_detection(img, threshold1, threshold2):
    """Finds edges in an image using the Canny algorithm.

    Parameters
    ----------
    img: ndarray
        the input image
    threshold1: int
        first threshold for the hysteresis procedure.
    threshold2: int
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

    h = img.shape[0]
    w = img.shape[1]

    mask = np.zeros((h, w), dtype=np.uint8)

    length = np.max((h, w))

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

            length = 40000

            pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
            pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))

        cv2.line(mask, pt1, pt2, color_value, 2, cv2.LINE_AA)  # cv2.LINE_AA

    return mask


def find_corners(img, max_number_corners=4, corner_quality=0.001, min_distance=20):
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
        Returns a NumPy array of the most prominent corners in the image, in the
        form (x,y).

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


def mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level):
    """Groups pixels together by colour and location.

    This function takes an image and mean-shift parameters and returns a version
    of the image that has had mean shift segmentation performed on it.

    Mean shift segmentation clusters nearby pixels with similar pixel values and
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


def find_largest_segment(img, color_difference=1, x_samples=8):
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
    x_samples: int
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
            if mask[y + 1, x + 1] == 0:
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


def create_segmented_image(img, contours):
    """
    Create an image the the contours are white filled and the rest is black.

    Parameters
    ----------
    img: ndarray
        input image necessary to the shape
    contours: list
        list of contours. Each individual contour is a Numpy array
        of (x,y) coordinates of boundary points of the object.

    Returns
    -------
    ndarray
        the segmented image
    """

    h = img.shape[0]
    w = img.shape[1]

    segmented = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(segmented, contours, -1, 255, cv2.FILLED)

    return segmented


def image_resize(img, width, height, interpolation=cv2.INTER_CUBIC):
    """Resize the input image to the given size.

    Resize the input image to the given size using the given interpolation
    method.

    Parameters
    ----------
    img: ndarray
        the input image
    width: int
        width of the target resized image
    height: int
        height of the target resized image
    interpolation: int
        interpolation method

    Returns
    -------
    tuple
        (ndarray, float) = (the resized image, the scale factor)
    """

    scale_factor = 1.
    resized_img = img
    h_img, w_img, c_img = img.shape

    if h_img > height and w_img > width:
        scale_factor = h_img / height
        height_scaled = height
        width_scaled = width
        resized_img = cv2.resize(img, (width_scaled, height_scaled), interpolation)

    return resized_img, scale_factor
