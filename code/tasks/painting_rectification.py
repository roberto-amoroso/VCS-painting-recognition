"""
Module containing functions to perform Painting Rectification.
"""

import cv2
import numpy as np

from utils.draw import show_image_window_blocking


def rectify_painting(src_img, corners, dst_img=None):
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
    src_points = corners
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
    img_rectified = cv2.warpPerspective(src_img, retval, (w_dst, h_dst), cv2.RANSAC)

    # show_image_window_blocking("rect_src", src_img)
    # print("src_shape: ", src_img.shape)
    # show_image_window_blocking("rect_dst", img_rectified)
    # print("dst_shape: ", img_rectified.shape)

    return img_rectified
