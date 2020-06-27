#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to execute the following operation, depending on the arguments received:
    - Painting Detection
    - Painting Segmentation
    - Painting Rectification
    - Painting Retrieval
    - People Detection
    - People Localization
"""
from model.task import Task
from model.media_type import MediaType
from pipeline_manager import PipelineManager
from utils import check_media_file, create_output_dir, create_directory
from draw import step_generator, draw_paintings_info, draw_people_bounding_box, show_image_window, print_next_step_info, \
    print_nicer, print_time_info
from image_processing import create_segmented_image, image_resize
from math_utils import translate_points
from painting_detection import detect_paintings
from painting_rectification import rectify_painting
from painting_retrieval import create_paintings_db, retrieve_paintings
from people_detection import detect_people
from paintings_and_people_localization import locale_paintings_and_people
from yolo.people_detection import PeopleDetection

import cv2
import os
import ntpath
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys


def main(args):
    pass


if __name__ == '__main__':

    # ------------------------------------------------------------------------------
    # Argument Parser
    # ---------------
    # TODO: manage using arguments
    # parser = argparse.ArgumentParser(
    #     description="Main script to execute the following operation, depending on the arguments received:\n"
    #                 "- Painting Detection\n"
    #                 "- Painting Rectification\n"
    #                 "- Painting Retrieval\n"
    #                 "- People Detection\n"
    #                 "- People Localization\n",
    #     formatter_class=argparse.RawTextHelpFormatter
    # )
    # parser.add_argument(
    #     "input_filename",
    #     help="filename of the input image or video"
    # )
    # parser.add_argument(
    #     "-vp",
    #     "--verbosity_print",
    #     type=int,
    #     choices=[0, 1],
    #     default=0,
    #     help="set the verbosity of the information displayed (description of "
    #          "the operation executed and execution time)\n"
    #          "\t0 = ONLY main processing steps info\n"
    #          "\t1 = ALL processing steps info"
    # )
    # parser.add_argument(
    #     "-vi",
    #     "--verbosity_image",
    #     type=int,
    #     choices=[0, 1, 2],
    #     default=0,
    #     help="set the verbosity of the images displayed\n"
    #          "NOTE: if the input is a video, is automatically set to '0'\n"
    #          "(in order to avoid an excessive number of images displayed on the screen).\n"
    #          "\t0 = NO image\n"
    #          "\t1 = ONLY main steps final images\n"
    #          "\t2 = ALL intermediate steps images"
    # )
    #
    # args = parser.parse_args()
    # main(args)
    #
    # ----- MAIN FUNCTION -----
    #
    # verbosity_print = args.verbosity_print
    # verbosity_image = args.verbosity_image

    verbosity_print = 0
    verbosity_image = 1
    task = Task(3)
    resize = True
    match_db_image = False
    histo_mode = False
    occurrence = 10
    # ----------------------
    # TESTING TODO remove it
    # ----------------------
    # input_filename = args.input_filename
    # input_filename = "dataset/photos/test/
    # VIDEOS
    input_filename = "dataset/videos/002/"
    # input_filename += "VID_20180529_112627.mp4"
    input_filename += "20180206_114604.mp4"
    # IMAGES
    # input_filename += 'IMG_2659_0012.jpg'  # CRITIC
    # input_filename += 'VID_20180529_113001_0000.jpg'  # LOTS painting not recognized
    # input_filename += "VID_20180529_112553_0002.jpg"  # Wall inverted
    # input_filename += "VID_20180529_112739_0004.jpg"  # Wall inverted
    # input_filename += "VID_20180529_112627_0000.jpg"  # Wall correct
    # input_filename += "VID_20180529_112517_0002.jpg"  # strange case
    # input_filename += "IMG_2646_0003.jpg"  # overlapping contours
    # input_filename += "IMG_2646_0006.jpg"  # overlapping contours
    # input_filename += "20180206_114604_0000.jpg"  # people
    # input_filename += "VID_20180529_112553_0004.jpg"  # wall inverted and cutted painting
    # input_filename += "IMG_2646_0018.jpg"  # wall inverted and cutted painting
    # input_filename += "VID_20180529_112553_0003.jpg"  # painting with error people detection

    # ------------------------------------------------------------------------------
    # Script execution time
    # ---------------------
    script_time_start = time.time()
    total_time = 0

    if occurrence < 1:
        sys.exit("occurrence should be >= 1\n")

    if resize:
        resize_height = 720
        resize_width = 1280
    else:
        # None means no resizing
        resize_height = None
        resize_width = None

    # ------------------------------------------------------------------------------
    # Check if input file is valid
    # -----------------------------
    media, media_type = check_media_file(input_filename)
    if media_type == MediaType.video:
        verbosity_image = 0

    # ------------------------------------------------------------------------------
    # Managing verbosity
    # ---------------

    if verbosity_print >= 1:
        print_next_step = print_next_step_info
        print_time = print_time_info
    else:
        print_next_step = print_time = lambda *a, **k: None

    if verbosity_image >= 2:
        show_image_main = show_image = show_image_window
    elif verbosity_image >= 1:
        show_image_main = show_image_window
        show_image = lambda *a, **k: None
    else:
        show_image_main = show_image = lambda *a, **k: None

    # ------------------------------------------------------------------------------
    # Print a summary of the invocation arguments
    # -----------------------------

    print()
    print("-" * 50)
    print("# SCRIPT CONFIGURATION:")
    print(f"\t-Task: {task.name}")
    print(f"\t-Filename: {input_filename}")
    print(f"\t-Media_type: {media_type.name}")
    print(f"\t-Verbosity_print: {verbosity_print}")
    print(f"\t-Verbosity_image: {verbosity_image}")
    if media_type == MediaType.video:
        if occurrence > 1:
            print(f"\t-Saving 1 frame every: {occurrence}")
        else:
            print('\t-Saving all frames')

    print("-" * 50)

    # ------------------------------------------------------------------------------
    # Instantiating general Objects
    # -----------------------------

    # Generator to keep track of the current step number
    generator = step_generator()

    # YOLO People Detector
    if task.value >= Task.people_detection.value:
        print("\n# Creating YOLO People Detector")
        people_detector = PeopleDetection()
    else:
        people_detector = None

    # ------------------------------------------------------------------------------
    # Instantiating DB
    # -----------------------------

    # DB path info
    painting_db_path = "./paintings_db"
    painting_data_path = "./data/data.csv"
    paintings_db = create_paintings_db(painting_db_path, painting_data_path)

    # ------------------------------------------------------------------------------
    # Instantiating output path
    # -----------------------------

    # Output path info
    output_base_path = "output/"
    output_path = os.path.join(output_base_path, task.name)

    if task == Task.painting_rectification:
        output_path = os.path.join(output_path, ntpath.basename(input_filename).split('.')[0])

    create_directory(output_path)

    print("\n\n")
    print("--------------------------------------------------")
    print("--------------   START PROCESSING   --------------")
    print("--------------------------------------------------")

    pipeline_manager = PipelineManager(
        input_filename,
        output_path,
        task,
        media_type,
        paintings_db,
        people_detector,
        resize_height,
        resize_width,
        match_db_image,
        histo_mode,
        generator,
        show_image_main,
        show_image,
        print_next_step,
        print_time
    )

    if media_type == MediaType.image:
        img_original = media

        print()
        print("-" * 50)
        print("# IMAGE INFO:")
        print(f"\t-Height: {img_original.shape[0]}")
        print(f"\t-Width: {img_original.shape[1]}")
        print("-" * 50)

        pipeline_manager.run(img_original)
    else:
        current_frame = 0
        frame_number = 0
        videoCapture = media
        frame_count = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        in_fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frame_process = int(frame_count // occurrence)
        # In order to have an output video of the same duration as the input one
        out_fps = (frame_process / frame_count) * in_fps
        duration = frame_count / in_fps

        print()
        print("-" * 50)
        print("# VIDEO INFO:")
        print('\t-Duration: {:.2f} s'.format(duration))
        print(f"\t-Frame count: {int(frame_count)}")
        print(f"\t-Frames to process: {frame_process}")
        print("\t-Input FPS: {:.2f}".format(in_fps))
        print("\t-Output FPS: {:.2f}".format(out_fps))
        print("-" * 50)

        success, img_original = videoCapture.read()
        if not success:
            sys.exit("Error while processing video frames.\n")
        height = img_original.shape[0]
        width = img_original.shape[1]
        filename, ext = pipeline_manager.out_filename.split('.')

        # Credits: https://github.com/ContinuumIO/anaconda-issues/issues/223#issuecomment-285523938
        if task != Task.painting_rectification:
            video = cv2.VideoWriter(
                os.path.join(output_path, '.'.join([filename, 'mp4'])),
                cv2.VideoWriter_fourcc(*"mp4v"),
                out_fps,
                (width, height)
            )
        while success and current_frame < frame_process:

            print()
            print("=" * 50)
            print(f"# PROCESSING FRAME #{current_frame + 1}/{frame_process}")
            print("=" * 50)
            current_filename = "_".join([filename, "{:05d}".format(current_frame)])
            pipeline_manager.out_filename = '.'.join([current_filename, "png"])

            # Process current frame
            img_original = pipeline_manager.run(img_original)

            # Write elaborated frame to create a video
            if task != Task.painting_rectification:
                video.write(img_original)

            current_frame += 1
            frame_number += occurrence
            videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, img_original = videoCapture.read()

        if current_frame != frame_process:
            sys.exit("Error while processing video frames.\n")
        if task != Task.painting_rectification:
            video.release()

    print()
    print("===================   RESULT   ===================")
    print("# Total execution time: {:.4f} s".format(time.time() - script_time_start))
    print("=" * 50)

    print("\n\n")
    print("--------------------------------------------------")
    print("---------------   END PROCESSING   ---------------")
    print("--------------------------------------------------")

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
