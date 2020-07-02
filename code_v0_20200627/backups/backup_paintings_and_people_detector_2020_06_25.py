#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to execute the following operation, depending on the arguments received:
    - Painting Detection
    - Painting Rectification
    - Painting Retrieval
    - People Detection
    - People Localization
"""
from utils.draw import step_generator, draw_paintings_info, draw_people_bounding_box, show_image_window, print_next_step_info, \
    print_nicer, print_time_info
from tasks.image_processing import create_segmented_image, image_resize
from utils.math_utils import translate_points
from tasks.painting_detection import detect_paintings
from tasks.painting_rectification import rectify_painting
from tasks.painting_retrieval import create_paintings_db, retrieve_paintings
from tasks.people_detection import detect_people
from tasks.paintings_and_people_localization import locale_paintings_and_people
from yolo.people_detection import PeopleDetection

import cv2
import os
import time
import numpy as np

if __name__ == '__main__':

    # ------------------------------------------------------------------------------
    # Script execution time
    # ---------------------
    script_time_start = time.time()
    total_time = 0

    # ------------------------------------------------------------------------------
    # Argument Parser
    # ---------------
    # TODO: manage using arguments
    """
    Verbosity_print:
    -v = 0 -> print only main script info (descr, time) [default]
    -v = 1 -> print also functions info (descr, time)  
    """
    verbosity_print = 0
    if verbosity_print >= 1:
        print_next_step = print_next_step_info
        print_time = print_time_info
    else:
        print_next_step = print_time = lambda *a, **k: None

    """ 
    Verbosity_image:
    -v = 0 -> show no images [default]
    -v = 1 -> show only main script images
    -v = 2 -> show also functions images 
    """
    verbosity_image = 2
    if verbosity_image >= 2:
        show_image_main = show_image = show_image_window
    elif verbosity_image >= 1:
        show_image_main = show_image_window
        show_image = lambda *a, **k: None
    else:
        show_image_main = show_image = lambda *a, **k: None

    # ------------------------------------------------------------------------------
    # Instantiating general Objects
    # -----------------------------

    # Generator to keep track of the current step number
    generator = step_generator()

    # YOLO People Detector
    people_detector = PeopleDetection()

    # ------------------------------------------------------------------------------
    # Instantiating general Objects
    # -----------------------------

    # DB path info
    painting_db_path = "./paintings_db"
    painting_data_path = "./data/data.csv"
    paintings_db = create_paintings_db(painting_db_path, painting_data_path)

    # Photo path info
    photos_path = 'dataset/photos'
    recognized_painting_path = 'dataset/recognized_paintings'
    videos_dir_name = 'test'  # 'test' or '013' or '009' or '014'
    filename = None
    # filename = 'IMG_2659_0012.jpg'  # CRITIC
    # filename = 'VID_20180529_113001_0000.jpg'  # LOTS painting not recognized
    # filename = "VID_20180529_112553_0002.jpg"  # Wall inverted
    # filename = "VID_20180529_112739_0004.jpg"  # Wall inverted
    # filename = "VID_20180529_112627_0000.jpg"  # Wall correct
    # filename = "VID_20180529_112517_0002.jpg"  # strange case
    # filename = "IMG_2646_0003.jpg" # overlapping contours
    # filename = "IMG_2646_0006.jpg"  # overlapping contours
    # filename = "20180206_114604_0000.jpg"  # people
    # filename = "VID_20180529_112553_0004.jpg"  # wall inverted and cutted painting
    # filename = "IMG_2646_0018.jpg"  # wall inverted and cutted painting
    # filename = "VID_20180529_112553_0003.jpg"  # painting with error people detection

    dst_dir = os.path.join(recognized_painting_path, videos_dir_name, "resize")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print('Created the directory "{}"'.format(dst_dir))
    else:
        print('The directory "{}" already exists'.format(dst_dir))

    for subdir, dirs, files in os.walk(os.path.join(photos_path, videos_dir_name)):
        for photo in files:
            if filename is not None:
                img_path = os.path.join(photos_path, videos_dir_name, filename)
            else:
                img_path = os.path.join(subdir, photo)

            print_nicer(f"PROCESSING IMAGE: {img_path}")

            img_original = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # ---------------------------------
            # RESIZING: resize to work only with HD resolution images
            # WARNINGS: if apply resizing, put color_differenze = 1 in largest segment
            # ---------------------------------
            print_nicer("RESIZE IMAGE - START")
            start_time = time.time()
            height = 720
            width = 1280
            img, scale_factor = image_resize(
                img_original,
                width,
                height,
                cv2.INTER_CUBIC
            )
            print_time_info(start_time, "RESIZE IMAGE - END")
            print(f"\tResized image shape: {img.shape}")
            show_image_main('image_resized', img, height=405, width=720)

            # ----------------------
            # PAINTING DETECTION
            # ----------------------
            print_nicer("PAINTING DETECTION - START")
            start_time = time.time()
            paintings_detected = detect_paintings(
                img,
                generator,
                show_image,
                print_next_step,
                print_time,
                scale_factor=scale_factor
            )
            print_time_info(start_time, "PAINTING DETECTION - END")

            # ----------------------
            # PAINTING SEGMENTATION
            # ----------------------
            print_nicer("PAINTING SEGMENTATION - START")
            painting_contours = [p.frame_contour for p in paintings_detected]
            segmented_img_original = create_segmented_image(
                img_original,
                painting_contours
            )
            print_time_info(start_time, "PAINTING SEGMENTATION - END")
            print("\tSegmented shape: ", segmented_img_original.shape)
            show_image_main('segmented_img_original', segmented_img_original, height=405, width=720)

            # ----------------------
            # PAINTING RECTIFICATION
            # ----------------------
            print_nicer("PAINTING RECTIFICATION - START")
            start_time = time.time()
            for i, painting in enumerate(paintings_detected):
                x, y, w_rect, h_rect = painting.bounding_box
                sub_img_original = img_original[y:y + h_rect, x:x + w_rect]
                corners = translate_points(painting.corners, -np.array([x, y]))
                painting.image = rectify_painting(sub_img_original, corners)
            print_time_info(start_time, "PAINTING RECTIFICATION - END")

            if show_image_main == show_image_window:
                for i, painting in enumerate(paintings_detected):
                    x, y, w_rect, h_rect = painting.bounding_box
                    sub_img_original = img_original[y:y + h_rect, x:x + w_rect]
                    show_image_main(f"sub_img_original_{i}", sub_img_original)
                    show_image_main(f"painting_rectified_{i}", painting.image)

            # ----------------------
            # PAINTING RETRIEVAL
            # ----------------------
            print_nicer("PAINTING RETRIEVAL - START")
            match_db_image = False
            start_time = time.time()
            retrieve_paintings(
                img,
                paintings_detected,
                paintings_db,
                generator,
                show_image,
                print_next_step,
                print_time,
                match_db_image=match_db_image
            )
            print_time_info(start_time, "PAINTING RETRIEVAL - END")

            if len(paintings_detected) > 0:

                # ----------------------
                # PEOPLE DETECTION
                # ----------------------
                print_nicer("PEOPLE DETECTION - START")
                start_time = time.time()
                max_percentage = 0.85
                people_bounding_boxes = detect_people(
                    img,
                    people_detector,
                    paintings_detected,
                    generator,
                    show_image,
                    print_next_step,
                    print_time,
                    scale_factor=scale_factor,
                    max_percentage=max_percentage
                )
                print_time_info(start_time, "PEOPLE DETECTION - END")

                # ----------------------
                # PEOPLE LOCALIZATION
                # ----------------------
                print_nicer("PEOPLE LOCALIZATION - START")
                start_time = time.time()
                people_room = locale_paintings_and_people(
                    paintings_detected,
                    generator,
                    show_image,
                    print_next_step,
                    print_time
                )
                print_time_info(start_time, "PEOPLE LOCALIZATION - END")

                # Step 18: Draw information about Paintings and People found
                # ----------------------------
                print_nicer("DRAW PAINTINGS AND PEOPLE INFORMATION - START")
                start_time = time.time()
                # First draw the info on the paintings...
                final_frame = draw_paintings_info(img_original, paintings_detected, people_room, scale_factor)
                # ...and then the info on the people, so as to respect the prospect.
                final_frame = draw_people_bounding_box(final_frame, people_bounding_boxes, scale_factor)
                print_time_info(start_time, "DRAW PAINTINGS AND PEOPLE INFORMATION - END")
            else:
                final_frame = img_original

            print("\n# Final frame shape: ", final_frame.shape)
            show_image_main('final_frame', final_frame, height=405, width=720)

            if filename is not None:
                break

            save_path = os.path.join(dst_dir, photo)
            cv2.imwrite(save_path, final_frame)

    print("\n", "=" * 30)
    print("# Total execution time: {:.4f} s".format(time.time() - script_time_start))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
