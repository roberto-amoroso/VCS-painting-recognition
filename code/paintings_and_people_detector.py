#!/usr/bin/env python
# -*- coding: utf-8 -*-

from draw import step_generator, print_next_step, show_image, draw_paintings_info, draw_people_bounding_box
from image_processing import create_segmented_image, image_resize
from math_utils import translate_points
from painting_detection import detect_paintings
from painting_rectification import rectify_painting
from painting_retrieval import create_paintings_db, retrieve_paintings
from people_detection_and_localization import detect_people
from yolo.people_detection import PeopleDetection

import cv2
import os
import time
import numpy as np

generator = step_generator()

if __name__ == '__main__':

    # YOLO People Detector
    people_detector = PeopleDetection()

    photos_path = 'dataset/photos'
    recognized_painting_path = 'dataset/recognized_paintings'
    videos_dir_name = 'test'  # 'test' or '013' or '009' or '014'
    filename = None
    # filename = '20180529_112417_ok_0031.jpg'
    # filename = '20180529_112417_ok_0026.jpg'
    # filename = 'IMG_2653_0002.jpg'
    # filename = 'IMG_2657_0006.jpg'
    # filename = 'IMG_2659_0012.jpg'  # CRITIC
    # filename = 'IMG_2659_0006.jpg'
    # filename = 'VID_20180529_113001_0000.jpg'  # LOTS painting not recognized
    # filename = "VID_20180529_112517_0004.jpg"
    # filename = "VID_20180529_112517_0005.jpg"
    # filename = "VID_20180529_112553_0002.jpg"  # Wall inverted
    # filename = "VID_20180529_112739_0004.jpg"  # Wall inverted
    # filename = "VID_20180529_112627_0000.jpg"  # Wall correct
    # filename = "VID_20180529_112517_0002.jpg"  # strange case
    # filename = "VID_20180529_112553_0005.jpg"
    # filename = "IMG_2646_0004.jpg"
    # filename = "IMG_2646_0003.jpg" # overlapping contours
    # filename = "IMG_2646_0006.jpg"  # overlapping contours
    # filename = "20180206_114604_0000.jpg"  # people
    # filename = "VID_20180529_112553_0004.jpg"  # wall inverted and cutted painting
    # videos_dir_name = '009'
    # filename = "IMG_2646_0018.jpg"  # wall inverted and cutted painting
    # filename = "VID_20180529_113001_0001.jpg"
    # filename = "VID_20180529_112553_0003.jpg"  # painting with error people detection
    # filename = "VID_20180529_112553_0000.jpg"
    # filename = "20180529_112417_ok_0004.jpg"
    # filename = "20180529_112417_ok_0004.jpg"
    # filename = "VID_20180529_112739_0007.jpg"
    # filename = "VID_20180529_112739_0007.jpg"
    # filename = "VIRB0402_0009.jpg"

    painting_db_path = "./paintings_db"
    painting_data_path = "./data/data.csv"

    paintings_db = create_paintings_db(painting_db_path, painting_data_path)
    total_time = 0

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

            print("\n#", "-" * 30)
            print(f"# Processing imag: {img_path}")
            print("#", "-" * 30, "\n")

            img_original = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # ---------------------------------
            # RESIZING: resize to work only with HD resolution images
            # WARNINGS: if apply resizing, put color_differenze = 1 in largest segment
            # ---------------------------------
            height = 720
            width = 1280
            img, scale_factor = image_resize(img_original, width, height, cv2.INTER_CUBIC)
            print(f"Image shape: {img.shape}")
            show_image('image_resized', img, height=405, width=720)

            # ----------------------
            # PAINTING DETECTION and SEGMENTATION
            # ----------------------
            paintings_detected = detect_paintings(img, generator)
            painting_contours = [p.frame_contour for p in paintings_detected]
            segmented_img_original = create_segmented_image(img_original, painting_contours, scale_factor)
            print("Segmented original shape: ", segmented_img_original.shape)
            show_image('segmented_img_original', segmented_img_original, height=405, width=720)

            # ----------------------
            # PAINTING RECTIFICATION
            # ----------------------
            for i, painting in enumerate(paintings_detected):
                x, y, w_rect, h_rect = np.int32(painting.bounding_box * scale_factor)
                sub_img_original = img_original[y:y + h_rect, x:x + w_rect]
                show_image(f"sub_img_original_{i}", sub_img_original)
                corners = np.int32(translate_points(painting.corners * scale_factor, -np.array([x, y])))
                painting.image = rectify_painting(sub_img_original, corners)
                show_image(f"painting_rectified_{i}", painting.image)

            # ----------------------
            # PAINTING RETRIEVAL
            # ----------------------
            retrieve_paintings(img, paintings_detected, paintings_db, generator)

            if len(paintings_detected) > 0:

                # ----------------------
                # PEOPLE DETECTION
                # ----------------------
                max_percentage = 0.85
                people_bounding_boxes = detect_people(
                    img,
                    people_detector,
                    paintings_detected,
                    generator,
                    max_percentage=max_percentage
                )

                # Step 18: Draw information about Paintings and People found
                # ----------------------------
                print_next_step(generator, "Draw paintings and people information:")
                start_time = time.time()
                # First draw the info on the paintings...
                final_frame = draw_paintings_info(img_original, paintings_detected, scale_factor)
                # ...and then the info on the people, so as to respect the prospect.
                final_frame = draw_people_bounding_box(final_frame, people_bounding_boxes, scale_factor)
                exe_time_draw_info = time.time() - start_time
                total_time += exe_time_draw_info
                print("\ttime: {:.3f} s".format(exe_time_draw_info))
                print("# Final frame shape: ", final_frame.shape)
                show_image('final_frame', final_frame, height=405, width=720)
                # cv2.imshow('final_frame', final_frame)
            else:
                final_frame = img_original

            if filename is not None:
                break

            save_path = os.path.join(dst_dir, photo)
            cv2.imwrite(save_path, final_frame)

    print()
    print("-" * 30)
    print("Total execution time: {:.3f} s".format(total_time))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
