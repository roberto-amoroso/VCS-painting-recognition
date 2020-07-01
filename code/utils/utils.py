"""
Module containing general utility functions.
"""
import sys
import cv2
import os
import ntpath

from models.media_type import MediaType


def create_directory(path):
    """
    Create directory at the given path, checking for errors and if the directory
    already exists.
    """
    path = ntpath.realpath(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(f"The syntax of the output file name, directory or volume is incorrect: {path}")
        else:
            print('\n# Created the output directory "{}"'.format(path))
    else:
        print('\n# The output directory "{}" already exists'.format(path))


def check_media_file(filename):
    """Check if the filename is related to a valid image or video.

    Parameters
    ----------
    filename: str
        name of the file to check

    Returns
    -------
    tuple or exit with error
        if filename is related to a valid media, returns it and its media type
        (0 = image, 1 = video). Otherwise, it exits with an error message

    """
    media_type = MediaType(0)
    filename = ntpath.realpath(filename)
    media = cv2.imread(filename, cv2.IMREAD_COLOR)
    if media is None:
        try:
            media_type = MediaType(1)
            media = cv2.VideoCapture(filename)
            if not media.isOpened():
                sys.exit(f"The input file should be a valid image or video: '{filename}'\n")
        except cv2.error as e:
            print("cv2.error:", e)
        except Exception as e:
            print("Exception:", e)
        # else:
        #     print("\n# VIDEO MODE - ON")
    # else:
    #     print("\n# IMAGE MODE - ON:")

    return media, media_type
