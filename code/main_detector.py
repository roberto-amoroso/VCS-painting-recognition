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
from models.task import Task
from models.media_type import MediaType
from models.pipeline_manager import PipelineManager
from utils.utils import check_media_file, create_directory
from utils.draw import step_generator, print_next_step_info, \
    print_nicer, print_time_info, show_image_window_blocking

from tasks.painting_retrieval import create_paintings_db
from yolo.people_detection import PeopleDetection

import cv2
import os
import ntpath
import time
import matplotlib.pyplot as plt
import sys
import argparse


def args_parse():
    # ---------------------------------------------------------------------------------------------
    # Argument Parser
    # ---------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Main script to execute the following tasks, depending on the value of the optional argument '-t' (or\n"
                    "'--task'):\n"
                    "- Painting Detection:               detects all paintings.\n"
                    "- Painting Segmentation:            creates a segmented version of the input, where the paintings,\n"
                    "                                    and also any statues, identified are white and the background\n"
                    "                                    is black.\n"
                    "- Painting Rectification:           rectifies each painting detected, through an affine transformation\n"
                    "- Painting Retrieval:               matches each detected and rectified painting to the paintings DB\n"
                    "                                    found in `db_path`\n"
                    "- People Detection:                 detects people in the input\n"
                    "- People and Painting Localization: locates paintings and people using information found in\n"
                    "                                    `data_filename`\n",
        epilog="# TASKS:\n"
               "\tThe tasks in the pipeline are performed in succession. Therefore, to execute the i-th task,\n"
               "\twith i>1, it is necessary that the j-th tasks are executed first, for each j such that 0<=j<i.\n"
               "\tFor example, if you want to perform Painting Rectification (i = 2) it is necessary that you\n"
               "\tfirst execute Painting Segmentation (j = 1) and Painting Detection (j = 0).\n"
               "\n"
               "\tThis implies that the Painting and People Detection task involves performing all the tasks.\n"
               "\tThis is also reflected in the output, which will be the union of the outputs of all the tasks\n"
               "\tin the pipeline.\n"
               "\n"
               "\tThe People Detection task is an exception. It runs independently of the other tasks.\n\n"
               "# OUTPUT:\n"
               "\tthe program output paths are structured as follows (let's consider '--output = \"output\"'):\n\n"
               "\t\toutput/\n"
               "\t\t |-- painting_detection/\n"
               "\t\t |-- painting_segmentation/\n"
               "\t\t |-- painting_rectification/\n"
               "\t\t |  |-- <input_filename>/\n"
               "\t\t |-- painting_retrieval/\n"
               "\t\t |-- people_detection/\n"
               "\t\t |-- paintings_and_people_localization/\n\n"
               "\tEach sub-directory will contain the output of the related task (indicated by the name of the\n"
               "\tsub-directory itself). The output filename will be the same of the input file processed.\n"
               "\tThe type of the output follows that of the input: 'image -> image' and 'video -> video'.\n"
               "\tThe exception is the Painting Rectification task, which produces only images as output,\n"
               "\tspecifically one image for each individual painting detected. It is clear that the number of \n"
               "\timages produced can be very high, especially in the case of videos. To improve the organization\n"
               "\tand access to data, the rectified images produced are stored in a directory that has the same\n"
               "\tas the input file processed. Inside this directory, the images are named as follows (the\n"
               "\textension is neglected):\n"
               "\t  input = image -> '<input_filename>_NN' where NN is a progressive number assigned to each\n"
               "\t                    painting found in the image.\n"
               "\t  input = video -> '<input_filename>_FFFFF_NN' where NN has the same meaning as before but\n"
               "\t                    applied to each video frame, while FFFFF is a progressive number assigned\n"
               "\t                    to each frame of the video that is processed.\n\n"
               "# FRAME_OCCURRENCE:\n"
               "\tin case '--frame_occurrence' is > 1, the frame rate of the output video will be set so that it\n"
               "\thas the same duration as the input video.\n\n"
               "# EXAMPLE:\n"
               "\tA full example could be:\n"
               "\t\"$ python main_detector.py dataset/videos/014/VID_20180529_112627.mp4"
               " painting_db/ data/data.csv -o output -t 5 -fo 30 -vp 1 -vi 1 --histo_mode\"\n\n"
               "\tIn this case, the input is a video and we want to perform the Painting and People Localization\n"
               "\ttask. This implies that all tasks (from 0 to 5) will be performed. The video will be processed \n"
               "\tconsidering one frame every 30 occurrences. All intermediate results will be printed, but no\n"
               "\timage will be displayed during processing because we are working with a video and '-vi' \n"
               "\tis automatically set equal to 0 (read '-vi' for details). The rectification of each detected\n"
               "\tpainting will be carried out only one time (better performance).\n"
               "\tIf ORB does not produce any match, a match based on histogram will be executed. The output\n"
               "\tis a video stored in './output/paintings_and_people_localization/VID_20180529_112627.mp4' whose\n"
               "\tframes show the results of the tasks performed on the frames of the input video.\n\n",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "input",
        type=str,
        help="image or video filename or folder containing images, videos or a mix of both\n\n"
    )

    parser.add_argument(
        "db_path",
        type=str,
        help="path of the directory where the images that make up the DB are located\n\n"
    )

    parser.add_argument(
        "data_filename",
        type=str,
        help="file containing all the information about the paintings:\n"
             "(Title, Author, Room, Image)\n\n"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="path used as base to determine where the outputs are stored. For details,\n"
             "read the epilogue at the bottom, section '# OUTPUT' \n\n"
    )

    parser.add_argument(
        "-t",
        "--task",
        type=int,
        choices=list(range(6)),
        default=5,
        help="determines which task will be performed on the input.\n"
             "NOTE: for details on how the tasks are performed and for some examples, read\n"
             "the epilogue at the bottom of the page, section '# TASKS'\n"
             "  0 = Painting Detection\n"
             "  1 = Painting Segmentation\n"
             "  2 = Painting Rectification\n"
             "  3 = Painting Retrieval\n"
             "  4 = People Detection\n"
             "  5 = People and Paintings Localization (DEFAULT)\n\n"
    )

    parser.add_argument(
        "-fo",
        "--frame_occurrence",
        type=int,
        default=1,
        help="integer >=1 (default =1). In case the input is a video, it establishes with \n"
             "which occurrence to consider the frames of the video itself.\n"
             "Example: frame_occurrence = 30 (VALUE RECOMMENDED DURING TESTING) means that\n"
             "it considers one frame every 30.\n"
             "NOTE: for more details read the epilogue at the bottom of the page, section \n"
             "'# FRAME_OCCURRENCE'\n\n"
    )

    parser.add_argument(
        "-vp",
        "--verbosity_print",
        type=int,
        choices=[0, 1],
        default=0,
        help="set the verbosity of the information displayed (description of the operation\n"
             "executed and its execution time)\n"
             "  0 = ONLY main processing steps info (DEFAULT)\n"
             "  1 = ALL processing steps info\n\n"
    )

    parser.add_argument(
        "-vi",
        "--verbosity_image",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="set the verbosity of the images displayed.\n"
             "NOTE: if the input is a video, is automatically set to '0' (in order to avoid\n"
             "an excessive number of images displayed on the screen).\n"
             "  0 = no image shown during processing (DEFAULT)\n"
             "  1 = shows output images of the main pipeline steps when they are created.\n"
             "      A button must be pressed to continue the execution (used for DEBUGGING)\n"
             "  2 = like '-v1 1', but it also shows output images of the intermediate\n"
             "      pipeline steps.\n\n"
    )

    parser.add_argument(
        "-mdbi",
        "--match_db_image",
        action="store_true",
        help="if present, to perform Painting Retrieval, the program rectifies each painting\n"
             "to match the aspect ration of every painting in 'db_path'. Otherwise, it\n"
             "rectifies each painting one time using a calculated aspect ratio.\n"
             "WARNING: setting '--match_db_image' slows down image processing as it\n"
             "introduces a new rectification operation for each painting in the DB.\n\n"
    )

    parser.add_argument(
        "-hm",
        "--histo_mode",
        action="store_true",
        help="if present indicates that, during Painting Retrieval, the program will executes\n"
             "Histogram Matching in the case ORB does not produce any match.\n"
             "WARNINGS: setting '--histo_mode' increases the percentage of matches with the \n"
             "DB, but decreases the precision, i.e. increases the number of false positives\n"
             "(incorrect matches).\n\n"
    )

    return parser.parse_args()


def main():
    """
    Main function executing the pipeline necessary to perform the main tasks.
    It receives the arguments from `ArgumentParser`
    """

    args = args_parse()

    input_filename = args.input
    painting_db_path = args.db_path
    painting_data_path = args.data_filename
    output_base_path = args.output
    task = Task(args.task)
    frame_occurrence = args.frame_occurrence
    verbosity_print = args.verbosity_print
    verbosity_image = args.verbosity_image
    match_db_image = args.match_db_image
    histo_mode = args.histo_mode

    # ---------------------------------------------------------------------------------------------
    # Script execution time
    # ---------------------------------------------------------------------------------------------
    script_time_start = time.time()

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

    # ---------------------------------------------------------------------------------------------
    # Check if input is valid
    # ---------------------------------------------------------------------------------------------

    try:
        inputs_list = [ntpath.join(ntpath.realpath('.'), input_filename, file) for file in os.listdir(input_filename)]
        inputs_list = [f for f in inputs_list if os.path.isfile(f)]
        if len(inputs_list) == 0:
            sys.exit(f"No images or video inside directory: {input_filename}")
    except NotADirectoryError:
        inputs_list = []
        inputs_list.append(ntpath.join(ntpath.realpath('.'), input_filename))
    except FileNotFoundError:
        sys.exit("No file or directory with the name {}".format(input_filename))

    # ---------------------------------------------------------------------------------------------
    # Check if DB path and Data filename are valid
    # ---------------------------------------------------------------------------------------------

    if not os.path.isdir(painting_db_path):
        sys.exit(f"\nError in DB path, should be a directory: '{painting_db_path}'\n")
    if not os.path.exists(painting_data_path):
        sys.exit(f"\nError in Data file: '{painting_data_path}'\n")

    # ---------------------------------------------------------------------------------------------
    # Instantiating output path
    # ---------------------------------------------------------------------------------------------

    # Output path info
    print_nicer("Creating output path")
    output_path = os.path.join(output_base_path, task.name)
    create_directory(output_path)
    print("-" * 50)

    # ---------------------------------------------------------------------------------------------
    # Managing prints verbosity
    # ---------------------------------------------------------------------------------------------

    if verbosity_print >= 1:
        print_next_step = print_next_step_info
        print_time = print_time_info
    else:
        print_next_step = print_time = lambda *a, **k: None

    # ---------------------------------------------------------------------------------------------
    # Print a summary of the invocation arguments
    # ---------------------------------------------------------------------------------------------

    print()
    print("-" * 50)
    print("# SCRIPT CONFIGURATION:")
    print("\t{:25s} {}".format("-Task:", task.name))
    print("\t{:25s} {}".format("-Input:", input_filename))
    if len(inputs_list) > 1:
        print("\t{:25s} {}".format("-Num input files:", len(inputs_list)))
    print("\t{:25s} {}".format("-DB path:", painting_db_path))
    print("\t{:25s} {}".format("-Data path:", painting_data_path))
    print("\t{:25s} {}".format("-Output base path:", output_base_path))
    print("\t{:25s} {}".format("-Verbosity_print:", verbosity_print))
    print("\t{:25s} {}".format("-Verbosity_image:", verbosity_image))
    print("\t{:25s} {}".format("-Task:", task.name))
    print("\t{:25s} {}".format("-Histo mode:", histo_mode))
    print("\t{:25s} {}".format("-Match DB images:", match_db_image))

    if frame_occurrence > 1:
        print("\t{:25s} {}".format("-Saving 1 frame every:", frame_occurrence))
    else:
        print("\t{:25s}".format("-Saving all video frames"))

    print("-" * 50)

    # ---------------------------------------------------------------------------------------------
    # Instantiating general Objects
    # ---------------------------------------------------------------------------------------------

    # Generator to keep track of the current step number
    generator = step_generator()

    # ---------------------------------------------------------------------------------------------
    # Instantiating DB
    # ---------------------------------------------------------------------------------------------
    if task == Task.painting_retrieval or task == Task.paintings_and_people_localization:
        # DB path info
        print_nicer('Loading paintings from DB')
        start_time = time.time()
        paintings_db = create_paintings_db(painting_db_path, painting_data_path)
        print(f"\tPaintings loaded:  {len(paintings_db)}")
        print_time_info(start_time)
        print("-" * 50)
    else:
        paintings_db = []

    # ---------------------------------------------------------------------------------------------
    # Instantiating YOLO People Detector
    # ---------------------------------------------------------------------------------------------
    if task.value >= Task.people_detection.value:
        print_nicer("Creating YOLO People Detector")
        start_time = time.time()
        people_detector = PeopleDetection()
        print_time_info(start_time)
        print("-" * 50)
    else:
        people_detector = None

    print("\n\n")
    print("--------------------------------------------------")
    print("--------------   START PROCESSING   --------------")
    print("--------------------------------------------------")

    verbosity_image_backup = verbosity_image
    tot_paintings_detected = 0
    tot_paintings_retrieved = 0
    tot_people_detected = 0

    for counter, input_filename in enumerate(inputs_list):

        if len(inputs_list) > 1:
            print()
            print("#" * 50)
            print(f"# PROCESSING FILE #{counter + 1}/{len(inputs_list)}")
            print("#" * 50)

        file_time_start = time.time()

        media, media_type = check_media_file(input_filename)
        if media_type == MediaType.video:
            verbosity_image = 0
        else:
            verbosity_image = verbosity_image_backup

        # ---------------------------------------------------------------------------------------------
        # Managing images verbosity
        # ---------------------------------------------------------------------------------------------

        if verbosity_image >= 2:
            show_image_main = show_image = show_image_window_blocking
        elif verbosity_image >= 1:
            show_image_main = show_image_window_blocking
            show_image = lambda *a, **k: None
        else:
            show_image_main = show_image = lambda *a, **k: None

        # ---------------------------------------------------------------------------------------------
        # Managing output information
        # ---------------------------------------------------------------------------------------------

        if task == Task.painting_rectification:
            out_path_info = output_path = os.path.join(output_path, ntpath.basename(input_filename).split('.')[0])
            create_directory(output_path)
        else:
            filename, ext = ntpath.basename(input_filename).split('.')
            ext = "." + ext if media_type == MediaType.image else ".mp4"
            out_path_info = os.path.join(output_path, filename + ext)

        out_path_info = os.path.abspath(out_path_info).replace('\\', '/')

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
            print("\t{:10s} {}".format("-Filename:", input_filename.replace('\\', '/')))
            print("\t{:10s} {}".format("-Height:", img_original.shape[0]))
            print("\t{:10s} {}".format("-Width:", img_original.shape[1]))
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
            print("\t{:25s} {}".format("-Filename:", input_filename.replace('\\', '/')))
            print("\t{:25s} {:.2f} s".format("-Duration:", duration))
            print("\t{:25s} {}".format("-Frame count:", int(frame_count)))
            print("\t{:25s} {}".format("-Frames to process:", frame_process))
            print("\t{:25s} {:.2f}".format("-Input FPS:", in_fps))
            print("\t{:25s} {}".format("-Output FPS:", out_fps))
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
        print("{:25s} {:.4f} s".format("# Total execution time:", time.time() - file_time_start))
        print("{:25s} {}".format("# Output path:", out_path_info))

        if media_type == MediaType.video:
            print("{:25s} {}".format("# Total frames processed:", current_frame))
        if task != Task.people_detection:
            print("{:25s} {}".format("# Painting detections:", pipeline_manager.num_paintings_detected))
        if task.value > Task.painting_rectification.value and task != Task.people_detection:
            print("{:25s} {}".format("# Painting retrievals:", pipeline_manager.num_paintings_retrieved))
        if task.value > Task.painting_retrieval.value or task == Task.people_detection:
            print("{:25s} {}".format("# People detections:", pipeline_manager.num_people_detected))
        print("=" * 50)

        tot_paintings_detected += pipeline_manager.num_paintings_detected
        tot_paintings_retrieved += pipeline_manager.num_paintings_retrieved
        tot_people_detected += pipeline_manager.num_people_detected

        plt.show()
        cv2.destroyAllWindows()

    print("\n\n")
    print("--------------------------------------------------")
    print("---------------   END PROCESSING   ---------------")
    print("--------------------------------------------------")

    print()
    print("================   FINAL RESULT   ================")
    print("{:25s} {:.4f} s".format("# Total execution time:", time.time() - script_time_start))

    if len(inputs_list) > 1:
        print("{:25s} {}".format("# Num files processed:", len(inputs_list)))

    if task != Task.people_detection:
        print("{:25s} {}".format("# Painting detections:", tot_paintings_detected))
    if task.value > Task.painting_rectification.value and task != Task.people_detection:
        print("{:25s} {}".format("# Painting retrievals:", tot_paintings_retrieved))
    if task.value > Task.painting_retrieval.value or task == Task.people_detection:
        print("{:25s} {}".format("# People detections:", tot_people_detected))
    print("=" * 50)

    plt.show()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
