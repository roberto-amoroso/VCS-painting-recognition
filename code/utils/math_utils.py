"""
Module containing mathematical utility functions.
"""
import numpy as np


def calculate_polygon_area(points):
    """
    Calculates the area of a polygon given its points, ordered clockwise,
    using Shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula).
    For example, in the case of 4 points, they should be sorted as follows:
     top-left, top-right, bottom-right, bottom-left

    Parameters
    ----------
    points: ndarray
        a Numpy array of value (x, y)
    Returns
    -------
    float
        the are of the polygon
    """
    area = 0.
    if points is not None:
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def order_points(pts):
    """Order a list of coordinates.

    Order a list of coordinates in a way
    such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the
    bottom-right, and the fourth is the bottom-left.
    
    Parameters
    ----------
    pts: ndarray
        list of coordinates

    Returns
    -------
    ndarray
        Returns a list of ordered coordinates

    Notes
    -----
    Credits:
    - https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    - https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    """
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    # D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]

    # My version
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def translate_points(points, translation):
    """
    Returns the points translated according to translation.
    """
    return points + translation
