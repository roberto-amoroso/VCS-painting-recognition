"""
Module containing general utility functions.
"""
import sys
import cv2


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
    media_type = 0
    media = cv2.imread(filename, cv2.IMREAD_COLOR)
    if media is None:
        try:
            media_type = 1
            media = cv2.VideoCapture(filename)
            if not media.isOpened():
                sys.exit("The input file should be a valid image or video.\n")
        except cv2.error as e:
            print("cv2.error:", e)
        except Exception as e:
            print("Exception:", e)
        # else:
        #     print("\n# VIDEO MODE - ON")
    # else:
    #     print("\n# IMAGE MODE - ON:")

    return media, media_type
