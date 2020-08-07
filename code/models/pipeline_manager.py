"""
Class managing the order of execution of the functions necessary to perform:
    - Painting Detection
    - Painting Segmentation
    - Painting Rectification
    - Painting Retrieval
    - People Detection
    - People and Paintings Localization
"""
from models.media_type import MediaType
from utils.draw import draw_paintings_info, draw_people_bounding_box, print_nicer, print_time_info, \
    draw_room_info
from tasks.image_processing import create_segmented_image, image_resize
from utils.math_utils import translate_points
from tasks.painting_detection import detect_paintings
from tasks.painting_rectification import rectify_painting
from tasks.painting_retrieval import retrieve_paintings
from tasks.people_detection import detect_people
from tasks.paintings_and_people_localization import locale_paintings_and_people

import cv2
import os
import time
import numpy as np
from models.task import Task
import ntpath


class PipelineManager:
    """
    Manages the order of execution of the functions necessary to perform:
    - Painting Detection
    - Painting Segmentation
    - Painting Rectification
    - Painting Retrieval
    - People Detection
    - People and Paintings Localization
    """

    def __init__(self,
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
                 print_time):
        self.input_filename = input_filename
        self.media_type = media_type
        self.output_path = output_path
        self.histo_mode = histo_mode
        self.match_db_image = match_db_image
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.people_detector = people_detector
        self.paintings_db = paintings_db
        self.task = task
        self.generator = generator
        self.show_image_main = show_image_main
        self.show_image = show_image
        self.print_next_step = print_next_step
        self.print_time = print_time

        # Counter used to save rectified images
        self.counter = 0

        # Define output filename
        self.out_filename = ntpath.basename(self.input_filename)

        # ----------------------
        # Create some stats info
        # ----------------------
        self.num_paintings_detected = 0
        self.num_paintings_retrieved = 0
        self.num_people_detected = 0

    def save_image(self, img):
        """
        Save the given image(s). The destination depends on the task.
        """
        if self.task == Task.painting_rectification:
            filename, ext = self.out_filename.split('.')
            # if self.media_type == MediaType.video:
            #     ext = 'png'  # 'jpg'
            out_name = "_".join([filename, "{:02d}".format(self.counter)])
            out_name = '.'.join([out_name, ext])
            self.counter += 1
        else:
            out_name = self.out_filename

        save_path = os.path.join(self.output_path, out_name)
        cv2.imwrite(save_path, img)

    def __resize_image(self, img_original):
        """
        Resize the input image
        """
        print_nicer("RESIZE IMAGE - START")
        start_time = time.time()
        img, scale_factor = image_resize(
            img_original,
            self.resize_width,
            self.resize_height,
            cv2.INTER_CUBIC
        )
        print_time_info(start_time, "RESIZE IMAGE - END")
        print(f"\tResized image shape: {img.shape}")
        print(f"\tScale factor: {scale_factor}")
        print("-" * 50)
        self.show_image_main('pipeline_input', img, height=405, width=720)

        return img, scale_factor

    def __painting_detection(self, img, scale_factor):
        """
        Execute Painting Detection on the input image.
        """

        print_nicer("PAINTING DETECTION - START")
        start_time = time.time()
        paintings_detected = detect_paintings(
            img,
            self.generator,
            self.show_image,
            self.print_next_step,
            self.print_time,
            scale_factor=scale_factor
        )
        print_time_info(start_time, "PAINTING DETECTION - END")
        print("-" * 50)

        self.num_paintings_detected += len(paintings_detected)

        return paintings_detected

    def __painting_segmentation(self, img_original, paintings_detected):
        """
        Execute Painting Segmentation of the input image.
        """

        print_nicer("PAINTING SEGMENTATION - START")
        start_time = time.time()
        painting_contours = [p.frame_contour for p in paintings_detected]
        segmented_img_original = create_segmented_image(
            img_original,
            painting_contours
        )
        print_time_info(start_time, "PAINTING SEGMENTATION - END")
        print("  Segmented shape: ", segmented_img_original.shape)
        print("-" * 50)
        # self.show_image_main('segmented_img_original_size', segmented_img_original, height=405, width=720)

        return segmented_img_original

    def __painting_rectification(self, img_original, paintings_detected):
        """
        Execute Rectification of the painting received
        """
        print_nicer("PAINTING RECTIFICATION - START")
        start_time = time.time()
        for i, painting in enumerate(paintings_detected):
            x, y, w_rect, h_rect = painting.bounding_box
            sub_img_original = img_original[y:y + h_rect, x:x + w_rect]
            corners = translate_points(painting.corners, -np.array([x, y]))
            painting.image = rectify_painting(sub_img_original, corners)
        print_time_info(start_time, "PAINTING RECTIFICATION - END")
        print("-" * 50)

        if self.task == Task.painting_rectification:
            for i, painting in enumerate(paintings_detected):
                x, y, w_rect, h_rect = painting.bounding_box
                sub_img_original = img_original[y:y + h_rect, x:x + w_rect]
                # self.show_image_main(f"sub_img_original_{i}", sub_img_original)
                self.show_image_main(f"painting_rectified_{i}", painting.image)

    def __painting_retrieval(self, paintings_detected, scale_factor):
        """
        Execute Painting Retrieval on the input image.
        """
        print_nicer("PAINTING RETRIEVAL - START")
        start_time = time.time()
        retrieve_paintings(
            paintings_detected,
            self.paintings_db,
            self.generator,
            self.show_image,
            self.print_next_step,
            self.print_time,
            match_db_image=self.match_db_image,
            histo_mode=self.histo_mode,
            scale_factor=scale_factor
        )
        print_time_info(start_time, "PAINTING RETRIEVAL - END")
        print("-" * 50)

        for p in paintings_detected:
            if p.title is not None:
                self.num_paintings_retrieved += 1

    def __people_detection(self, img, paintings_detected, scale_factor):
        """
        Execute People Detection on the input image.
        """
        print_nicer("PEOPLE DETECTION - START")
        start_time = time.time()
        max_percentage = 0.85
        people_bounding_boxes = detect_people(
            img,
            self.people_detector,
            paintings_detected,
            self.generator,
            self.show_image,
            self.print_next_step,
            self.print_time,
            scale_factor=scale_factor,
            max_percentage=max_percentage
        )
        print_time_info(start_time, "PEOPLE DETECTION - END")
        print("-" * 50)

        self.num_people_detected += len(people_bounding_boxes)

        return people_bounding_boxes

    def __paintings_and_people_localization(self, paintings_detected):
        """
        Execute People Localization using Information about the detected paintings.
        """

        print_nicer("PEOPLE LOCALIZATION - START")
        start_time = time.time()
        people_room = locale_paintings_and_people(
            paintings_detected,
            self.generator,
            self.show_image,
            self.print_next_step,
            self.print_time
        )
        print_time_info(start_time, "PEOPLE LOCALIZATION - END")
        print("-" * 50)

        return people_room

    def __draw_painting_and_people(self, img_original, paintings_detected, people_bounding_boxes, people_room,
                                   scale_factor):
        """
        Draw information about Paintings and People found.
        """
        print_nicer("DRAW PAINTINGS AND PEOPLE INFORMATION - START")
        start_time = time.time()
        # First draw the info on the paintings...
        draw_paintings_info(img_original, paintings_detected, scale_factor)
        # ...and then the info on the people, so as to respect the prospect...
        draw_people_bounding_box(img_original, people_bounding_boxes, scale_factor)
        # ...at the end the information about the room
        draw_room_info(img_original, people_room, scale_factor)
        print_time_info(start_time, "DRAW PAINTINGS AND PEOPLE INFORMATION - END")
        print("-" * 50)

    def run(self, img_original):
        """
        It execute the `self.task` operation on single images, so in the case of videos,
        it should be called on every frame.
        """

        # ---------------------------------
        # RESIZING: resize to work only with HD resolution images
        # ---------------------------------
        if self.resize_height is not None and self.resize_width is not None:
            img, scale_factor = self.__resize_image(img_original)
        else:
            img = img_original
            scale_factor = 1

        # ------------------------------------------------------------------------
        paintings_detected = []
        if self.task != Task.people_detection:
            # ----------------------
            # PAINTING DETECTION
            # ----------------------
            paintings_detected = self.__painting_detection(img, scale_factor)

        # ------------------------------------------------------------------------
        if self.task.value > Task.painting_detection.value and self.task != Task.people_detection:
            # ----------------------
            # PAINTING SEGMENTATION
            # ----------------------
            segmented_img_original = self.__painting_segmentation(
                img_original,
                paintings_detected
            )

            if self.task == Task.painting_segmentation:
                img_original = segmented_img_original
                # Replicate over 3 channel to avoid problem with `cv2.VideoWriter()`
                if self.media_type == MediaType.video:
                    img_original = cv2.merge((img_original, img_original, img_original))
                paintings_detected = []

        # ------------------------------------------------------------------------

        if self.task.value > Task.painting_segmentation.value and self.task != Task.people_detection:
            # ----------------------
            # PAINTING RECTIFICATION
            # ----------------------
            self.__painting_rectification(img_original, paintings_detected)

            # Special case to accomplish mandatory task requirements
            if self.task == Task.painting_rectification:
                for self.counter, painting in enumerate(paintings_detected):
                    self.save_image(painting.image)

        # ------------------------------------------------------------------------

        if self.task.value > Task.painting_rectification.value and self.task != Task.people_detection:
            # ----------------------
            # PAINTING RETRIEVAL
            # ----------------------
            self.__painting_retrieval(paintings_detected, scale_factor)

        # ------------------------------------------------------------------------
        people_bounding_boxes = []

        if self.task.value > Task.painting_retrieval.value or self.task == Task.people_detection:
            # ----------------------
            # PEOPLE DETECTION
            # ----------------------
            people_bounding_boxes = self.__people_detection(
                img,
                paintings_detected,
                scale_factor
            )

        # ------------------------------------------------------------------------
        people_room = -1
        if self.task.value > Task.people_detection.value:
            # ----------------------
            # PAINTINGS AND PEOPLE LOCALIZATION
            # ----------------------
            people_room = self.__paintings_and_people_localization(paintings_detected)

            # If I want to execute only people localization, I will not show info
            # about paintings, otherwise I have `self.task == Task.all_in`
            # if self.task == Task.paintings_and_people_localization:
            #     paintings_detected = []

        # ------------------------------------------------------------------------

        # Step 18: Draw information about Paintings and People found
        # ----------------------------
        self.__draw_painting_and_people(
            img_original,
            paintings_detected,
            people_bounding_boxes,
            people_room,
            scale_factor
        )

        if self.task != Task.painting_rectification:
            # print("\n# Final frame shape: ", img_original.shape)
            self.show_image_main('pipeline_output_' + self.task.name, img_original, height=405, width=720)
            if self.media_type == MediaType.image:
                self.save_image(img_original)

        return img_original
