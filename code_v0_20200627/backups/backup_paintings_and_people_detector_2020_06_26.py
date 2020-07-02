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
from models.task import Task
from models.media_type import MediaType
from models.pipeline_manager import PipelineManager
from utils.utils import check_media_file, create_output_dir
from utils.draw import step_generator, show_image_window, print_next_step_info, \
    print_time_info
from tasks.painting_retrieval import create_paintings_db
from yolo.people_detection import PeopleDetection

import cv2
import os
import time
import matplotlib.pyplot as plt

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
    #
    # verbosity_print = args.verbosity_print
    # verbosity_image = args.verbosity_image

    verbosity_print = 0
    verbosity_image = 0
    task = Task(2)
    resize_height = 720
    resize_width = 1280
    match_db_image = False
    histo_mode = False
    # ----------------------
    # TESTING TODO remove it
    # ----------------------
    # input_filename = args.input_filename
    input_filename = "dataset/photos/test/"
    # input_filename += 'IMG_2659_0012.jpg'  # CRITIC
    # input_filename += 'VID_20180529_113001_0000.jpg'  # LOTS painting not recognized
    # input_filename += "VID_20180529_112553_0002.jpg"  # Wall inverted
    # input_filename += "VID_20180529_112739_0004.jpg"  # Wall inverted
    input_filename += "VID_20180529_112627_0000.jpg"  # Wall correct
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

    print(f"\t-Filename: {input_filename}")
    print(f"\t-Media_type: ", end='')
    if media_type == MediaType.image:
        print(f"{MediaType.image.name}")
    else:
        print(f"{MediaType.video.name}")
    print(f"\t-Verbosity_print: {verbosity_print}")
    print(f"\t-Verbosity_image: {verbosity_image}")

    print("-" * 50)

    # ------------------------------------------------------------------------------
    # Instantiating general Objects
    # -----------------------------

    # Generator to keep track of the current step number
    generator = step_generator()

    # YOLO People Detector
    if task == Task.people_detection:
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
    output_path = create_output_dir(output_base_path, task, media_type)

    output_filename = os.path.join(output_base_path, os.path.basename(input_filename))

    img_original = media

    print("\n\n")
    print("--------------------------------------------------")
    print("--------------   START PROCESSING   --------------")
    print("--------------------------------------------------")

    pipeline_manager = PipelineManager(
        task,
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

    pipeline_manager.run(img_original, input_filename)
    cv2.imwrite(output_filename, img_original)

    print()
    print("=" * 50)
    print("# Total execution time: {:.4f} s".format(time.time() - script_time_start))
    print("=" * 50)

    print("\n\n")
    print("--------------------------------------------------")
    print("---------------   END PROCESSING   ---------------")
    print("--------------------------------------------------")

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
