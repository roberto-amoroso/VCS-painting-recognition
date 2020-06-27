#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to execute the following tasks, depending on the arguments received:
    - Painting Detection
    - Painting Segmentation
    - Painting Rectification
    - Painting Retrieval
    - People Detection
    - People and Paintings Localization
"""
from model.task import Task
from model.media_type import MediaType
from pipeline_manager import PipelineManager
from utils import check_media_file, create_output_dir, create_directory
from draw import step_generator, draw_paintings_info, draw_people_bounding_box, show_image_window, print_next_step_info, \
    print_nicer, print_time_info, show_image_window_blocking
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
    """
    Main function executing the pipeline necessary to perform the main tasks.
    It receives the arguments from `ArgumentParser`
    """

    # input_filename = args.input_filename
    # painting_db_path = args.db_path
    # painting_data_path = args.data_filename
    # output_base_path = args.output
    # task = Task(args.task)
    # frame_occurrence = args.frame_occurrence
    # verbosity_print = args.verbosity_print
    # verbosity_image = args.verbosity_image
    # match_db_image = args.match_db_image
    # histo_mode = args.histo_mode

    verbosity_print = 1
    verbosity_image = 1
    task = Task(5)
    match_db_image = True
    histo_mode = True
    frame_occurrence = 30
    painting_db_path = "paintings_db/"
    painting_data_path = "data/data.csv"
    output_base_path = "output"
    # ----------------------
    # TESTING TODO remove it
    # ----------------------
    input_filename = "dataset/photos/test/"
    # VIDEOS
    # input_filename = "dataset/videos/002/"
    # input_filename += "VID_20180529_112627.mp4"
    # input_filename += "20180206_114604.mp4"
    # IMAGES
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

    if frame_occurrence < 1:
        sys.exit("frame_occurrence should be >= 1\n")

    # For extensibility and future improvements
    resize = True
    if resize:
        resize_height = 720
        resize_width = 1280
    else:
        # None means no resizing
        resize_height = None
        resize_width = None

    # ------------------------------------------------------------------------------
    # Check if DB path and Data filename are valid
    # -----------------------------
    if not os.path.isdir(painting_db_path):
        sys.exit(f"\nError in DB path: '{painting_db_path}'\n")
    if not os.path.exists(painting_data_path):
        sys.exit(f"\nError in Data file: '{painting_data_path}'\n")

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
        show_image_main = show_image = show_image_window_blocking
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
    print(f"\t-Task:                 {task.name}")
    print(f"\t-Filename:             {input_filename}")
    print(f"\t-Media_type:           {media_type.name}")
    print(f"\t-DB path:              {painting_db_path}")
    print(f"\t-Data path:            {painting_data_path}")
    print(f"\t-Output base path:     {output_base_path}")
    print(f"\t-Verbosity_print:      {verbosity_print}")
    print(f"\t-Verbosity_image:      {verbosity_image}")
    print(f"\t-Match DB images:      {match_db_image}")
    print(f"\t-Histo mode:           {histo_mode}")
    if media_type == MediaType.video:
        if frame_occurrence > 1:
            print(f"\t-Saving 1 frame every: {frame_occurrence}")
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
        print_nicer("Creating YOLO People Detector")
        start_time = time.time()
        people_detector = PeopleDetection()
        print_time(start_time)
        print("-" * 50)
    else:
        people_detector = None

    # ------------------------------------------------------------------------------
    # Instantiating DB
    # -----------------------------

    # DB path info
    paintings_db = create_paintings_db(painting_db_path, painting_data_path)

    # ------------------------------------------------------------------------------
    # Instantiating output path
    # -----------------------------

    # Output path info
    print_nicer("Creating output path")
    output_path = os.path.join(output_base_path, task.name)

    if task == Task.painting_rectification:
        output_path = os.path.join(output_path, ntpath.basename(input_filename).split('.')[0])

    create_directory(output_path)

    print("-" * 50)

    print("\n\n")
    print("--------------------------------------------------")
    print("--------------   START PROCESSING   --------------")
    print("--------------------------------------------------")

    pipeline_manager = PipelineManager(
        input_filename=input_filename,
        output_path=output_path,
        task=task,
        media_type=media_type,
        paintings_db=paintings_db,
        people_detector=people_detector,
        resize_height=resize_height,
        resize_width=resize_width,
        match_db_image=match_db_image,
        histo_mode=histo_mode,
        generator=generator,
        show_image_main=show_image_main,
        show_image=show_image,
        print_next_step=print_next_step,
        print_time=print_time
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
        frame_process = int(frame_count // frame_occurrence)
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
            frame_number += frame_occurrence
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


if __name__ == '__main__':
    # ------------------------------------------------------------------------------
    # Argument Parser
    # ---------------
    # parser = argparse.ArgumentParser(
    #     description="Main script to execute the following tasks, depending on the value of the optional argument '-t' (or\n"
    #                 "'--task'):\n"
    #                 "- Painting Detection:               detects all paintings.\n"
    #                 "- Painting Segmentation:            creates a segmented version of the input, where the paintings\n"
    #                 "                                    and statues identified are white and the background is black.\n"
    #                 "- Painting Rectification:           rectifies each painting detected, through an affine transformation\n"
    #                 "- Painting Retrieval:               matches each detected and rectified painting to the paintings DB\n"
    #                 "                                    found in `db_path`\n"
    #                 "- People Detection:                 detects people in the input\n"
    #                 "- People and Painting Localization: locates paintings and people using information found in\n"
    #                 "                                    `data_filename`\n",
    #     epilog="# TASKS:\n"
    #            "\tGiven the mutual dependency of the tasks, to execute the i-th task, with i>1, it is necessary\n"
    #            "\tthat the j-th tasks are executed first, for each j such that 0<=j<i.\n"
    #            "\tFor example, if you want to perform Painting Rectification (i = 2) it is necessary that you\n"
    #            "\tfirst execute Painting Segmentation (j = 1) and Painting Detection (j = 0).\n"
    #            "\tThe People Detection task is an exception. It runs independently of the other tasks.\n\n"
    #            "# OUTPUT:\n"
    #            "\tthe program output paths are structured as follows (let's consider '--output = \"output\"'):\n\n"
    #            "\t\toutput/\n"
    #            "\t\t |-- painting_detection/\n"
    #            "\t\t |-- painting_segmentation/\n"
    #            "\t\t |-- painting_rectification/\n"
    #            "\t\t |  |-- <input_filename>/\n"
    #            "\t\t |-- painting_retrieval/\n"
    #            "\t\t |-- people_detection/\n"
    #            "\t\t |-- paintings_and_people_localization/\n\n"
    #            "\tEach sub-directory will contain the output of the related task (indicated by the name of the\n"
    #            "\tsub-directory itself). The output filename will be the same of the input ('input_filename').\n"
    #            "\tThe type of the output follows that of the input: 'image -> image' and 'video -> video'.\n"
    #            "\tThe exception is the Painting Rectification task, which produces only images as output,\n"
    #            "\tspecifically one image for each individual painting detected. it is clear that the number of \n"
    #            "\timages produced can be very high, especially in the case of videos. To improve the organization\n"
    #            "\tand access to data, the rectified images produced are stored in a directory that has the same\n"
    #            "\tas 'input_filename'. Inside this directory, the images are named as follows (the extension is\n"
    #            "\tneglected):\n"
    #            "\t  input = image -> '<input_filename>_NN' where NN is a progressive number assigned to each\n"
    #            "\t                    painting found in the image.\n"
    #            "\t  input = video -> '<input_filename>_FFFFF_NN' where NN has the same meaning as before but\n"
    #            "\t                    applied to each video frame, while FFFFF is a progressive number assigned\n"
    #            "\t                    to each frame of the video that is processed.\n\n"
    #            "# FRAME_OCCURRENCE:\n"
    #            "\tin case '--frame_occurrence' is > 1, the frame rate of the output video will be set so that it\n"
    #            "\thas the same duration as the input video.\n\n"
    #            "# EXAMPLE:\n"
    #            "\tA full example could be:\n"
    #            "\t\"$ python paintings_and_people_detector.py dataset/videos/014/VID_20180529_112627.mp4"
    #            " painting_db/ data/data.csv -o output -t 5 -fo 30 -vp 1 -vi 2 --match_db_image --histo_mode\"\n\n"
    #            "\tEXPLANATION:\n"
    #            "\tIn this case, the input is a video and we want to perform the Painting and People Localization\n"
    #            "\ttask. This implies that all tasks (0 to 5) will be performed. The video will be processed \n"
    #            "\tconsidering one frame every 30 occurrences. All intermediate results will be printed, but no\n"
    #            "\timage will be displayed during processing because we are working with a video and '-vi' \n"
    #            "\tis automatically set equal to 0 (read '-vi' for details). The rectification of each detected\n"
    #            "\tpainting will be carried out to match the aspect ratio of each image of the db. In the event \n"
    #            "\tthat ORB does not produce any match, a match based on histogram will be executed. The output\n"
    #            "\tis a video stored in './output/paintings_and_people_localization/VID_20180529_112627.mp4' whose\n"
    #            "\tframes show the results of the various tasks performed.\n\n",
    #     formatter_class=argparse.RawTextHelpFormatter
    # )
    #
    # parser.add_argument(
    #     "input_filename",
    #     type=str,
    #     help="filename of the input image or video\n\n"
    # )
    #
    # parser.add_argument(
    #     "db_path",
    #     type=str,
    #     help="path of the directory where the images that make up the DB are located\n\n"
    # )
    #
    # parser.add_argument(
    #     "data_filename",
    #     type=str,
    #     help="file containing all the information about the paintings:\n"
    #          "(Title, Author, Room, Image)\n\n"
    # )
    #
    # parser.add_argument(
    #     "-o",
    #     "--output",
    #     type=str,
    #     default="output",
    #     help="path used as base to determine where the outputs are stored. For details,\n"
    #          "read the epilogue at the bottom, section '# OUTPUT' \n\n"
    # )
    #
    # parser.add_argument(
    #     "-t",
    #     "--task",
    #     type=int,
    #     choices=list(range(6)),
    #     default=5,
    #     help="determines which task will be performed on the input.\n"
    #          "NOTE: for details on how the tasks are performed and for some examples, read\n"
    #          "the epilogue at the bottom of the page, section '# TASKS'\n"
    #          "  0 = Painting Detection\n"
    #          "  1 = Painting Segmentation\n"
    #          "  2 = Painting Rectification\n"
    #          "  3 = Painting Retrieval\n"
    #          "  4 = People Detection\n"
    #          "  5 = People and Paintings Localization (default)\n\n"
    # )
    #
    # parser.add_argument(
    #     "-fo",
    #     "--frame_occurrence",
    #     type=int,
    #     default=1,
    #     help="integer >=1 (default =1). In case the input is a video, it establishes with \n"
    #          "which occurrence to consider the frames of the video itself.\n"
    #          "Example: frame_occurrence = 30 (value recommended during debugging) means that\n"
    #          "it considers one frame every 30.\n\n"
    # )
    #
    # parser.add_argument(
    #     "-vp",
    #     "--verbosity_print",
    #     type=int,
    #     choices=[0, 1],
    #     default=0,
    #     help="set the verbosity of the information displayed (description of the operation\n"
    #          "executed and its execution time)\n"
    #          "  0 = ONLY main processing steps info (default)\n"
    #          "  1 = ALL processing steps info\n\n"
    # )
    #
    # parser.add_argument(
    #     "-vi",
    #     "--verbosity_image",
    #     type=int,
    #     choices=[0, 1, 2],
    #     default=0,
    #     help="set the verbosity of the images displayed.\n"
    #          "NOTE: if the input is a video, is automatically set to '0' (in order to avoid\n"
    #          "an excessive number of images displayed on the screen).\n"
    #          "  0 = no image shown (default)\n"
    #          "  1 = shows main steps final images, at the end of the script execution\n"
    #          "      (NOT BLOCKING)\n"
    #          "  2 = shows each final and intermediate image when it is created and a button\n"
    #          "      must be pressed to continue the execution (mainly used for debugging)\n"
    #          "      (BLOCKING)\n\n"
    # )
    #
    # parser.add_argument(
    #     "-mdbi",
    #     "--match_db_image",
    #     action="store_true",
    #     help="if present, to perform Painting Retrieval, the program rectifies each painting\n"
    #          "to match the aspect ration of every painting in 'db_path'. Otherwise, it\n"
    #          "rectifies each painting one time using a calculated aspect ratio.\n\n"
    # )
    #
    # parser.add_argument(
    #     "-hm",
    #     "--histo_mode",
    #     action="store_true",
    #     help="if present indicates that, during Painting Retrieval, the program will executes\n"
    #          "Histogram Matching in the case ORB does not produce any match.\n\n"
    # )
    #
    # args = parser.parse_args()
    # main(args)
    main(None)
