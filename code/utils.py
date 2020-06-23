import cv2
import numpy as np
from scipy.spatial import distance as dist
import os


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
    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # if height is not None and width is not None:
    #     cv2.resizeWindow(title, width, height)
    # cv2.imshow(title, img)
    # if wait_key:
    #     cv2.waitKey(0)
    pass


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
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)


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


if __name__ == '__main__':
    # db = create_paintings_db("./paintings_db")
    # for num in step_generator("A"):
    #     if num > 5:
    #         break
    #     print(num)

    generator = step_generator()

    print_next_step(generator, "A")
    print_next_step(generator, "B")

    # dilated_wall_mask = image_dilation(wall_mask_inverted, 50)
    # show_image('test1', dilated_wall_mask)
    #
    # eroded_wall_mask = image_erosion(dilated_wall_mask, 60)
    # show_image('test2', eroded_wall_mask)
    #
    # eroded_wall_mask = image_morphology_tranformation(dilated_wall_mask, cv2.MORPH_OPEN, 60)
    # show_image('test3', eroded_wall_mask)
