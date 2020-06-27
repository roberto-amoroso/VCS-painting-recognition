"""
Module containing general utility functions.
"""
import sys
import cv2
import os
import ntpath

from model.media_type import MediaType
from model.task import Task


def create_rectification_output_dirs(base_path, dir_1="original", dir_2="rectified"):
    """Create output directory path where the results of the painting rectification
    process will be saved.

    Parameters
    ----------
    base_path: str
        base output path
    dir_1: str
        name of directory where original paintings images will be saved
    dir_2: str
        name of directory where rectified paintings images will be saved

    Returns
    -------
    tuple
        the paths of the created directory

    """

    output_path_1 = os.path.join(base_path, dir_1)
    output_path_2 = os.path.join(base_path, dir_2)

    create_directory(output_path_1)
    create_directory(output_path_2)

    return output_path_1, output_path_2


def create_output_dir(base_path, task):
    """Create output directory path where the results will be saved.

    Parameters
    ----------
    base_path:str
        base output path
    task: Task
        task to be executed

    Returns
    -------
    str or tuple
        output directory path(s)
    """
    output_path = os.path.join(base_path, task.name)

    if task == Task.painting_rectification:
        output_path = create_rectification_output_dirs(output_path)

    return output_path

# TODO: remove following comment
# def create_output_dir(base_path, media_type, task, input_filename):
#     """Create output directory path where the results will be saved.
#
#     Parameters
#     ----------
#     base_path:str
#         base output path
#     task: Task
#         task to be executed
#     media_type: MediaType
#         media type of the input
#     input_filename: str
#         name of the input file
#
#     Returns
#     -------
#     str or tuple
#         output directory path(s)
#     """
#
#     input_filename_only = ntpath.basename(input_filename).split('.')[0]
#     output_path = os.path.join(base_path, media_type.name + 's', task.name, input_filename)
#
#     if task == Task.painting_rectification:
#         output_path = create_rectification_output_dirs(output_path)
#
#     return output_path



def create_directory(path):
    """
    Create directory at the given path, checking for errors and if the directory
    already exists.
    """

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
    media = cv2.imread(filename, cv2.IMREAD_COLOR)
    if media is None:
        try:
            media_type = MediaType(1)
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
