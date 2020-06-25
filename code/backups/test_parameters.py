from image_processing import mean_shift_segmentation, find_largest_segment, image_dilation, image_erosion, invert_image
from math_utils import print_next_step, step_generator, show_image, draw_lines, draw_corners, order_points, translate_points, \
    calculate_polygon_area, draw_people_bounding_box
import cv2
import numpy as np
import os
import time

generator = step_generator()

if __name__ == '__main__':
    photos_path = '../dataset/photos'
    recognized_painting_path = '../dataset/recognized_paintings'
    videos_dir_name = 'test'
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


    painting_db_path = "../paintings_db"
    painting_data_path = "../data/data.csv"

    for spatial in range(7, 9):
        for color in range(11, 17, 1):
            for col_diff in range(2, 3):

                total_time = 0

                print("#", "-" * 30)
                print(f"# Testing:\n\tspatial: {spatial}\n\tcolor: {color}")
                print("#", "-" * 30)

                dst_dir = os.path.join(recognized_painting_path, videos_dir_name,
                                       f"spatial-{spatial}_color-{color}_coldiff-{col_diff}")
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

                        h_img, w_img, c_img = img_original.shape

                        scale_factor = h_img / height

                        height_scaled = np.min((h_img, height))
                        width_scaled = np.min((w_img, width))
                        img = cv2.resize(img_original, (width_scaled, height_scaled), cv2.INTER_CUBIC)

                        print(f"Image shape: {img.shape}")
                        show_image('image_original', img, height=405, width=720)

                        # Step 1: Perform mean shift segmentation on the image
                        # ----------------------------
                        print_next_step(generator, "Mean Shift Segmentation:")
                        start_time = time.time()
                        spatial_radius = spatial  # 8 # 5 #8 or 7
                        color_radius = color  # 40 #40 #35 or 15
                        maximum_pyramid_level = 1  # 1
                        img_mss = mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level)
                        exe_time_mean_shift_segmentation = time.time() - start_time
                        total_time += exe_time_mean_shift_segmentation
                        print("\ttime: {:.3f} s".format(exe_time_mean_shift_segmentation))
                        show_image('image_mean_shift_segmentation', img_mss, height=405, width=720)

                        # Step 2: Get a mask of just the wall in the gallery
                        # ----------------------------
                        print_next_step(generator, "Mask the Wall:")
                        start_time = time.time()
                        color_difference = col_diff  # 2 # 1
                        x_samples = 8  # 8 or 16
                        wall_mask = find_largest_segment(img_mss, color_difference, x_samples)
                        exe_time_mask_largest_segment = time.time() - start_time
                        total_time += exe_time_mask_largest_segment

                        print("\ttime: {:.3f} s".format(exe_time_mask_largest_segment))
                        show_image(f'image_mask_largest_segment', wall_mask, height=405,
                                   width=720)
                        cv2.imwrite(
                            os.path.join(dst_dir, photo.split('.')[0] + "_largest_segment." + photo.split('.')[1]),
                            wall_mask)

                        # Step 3: Dilate and Erode the wall mask to remove noise
                        # ----------------------------
                        # IMPORTANT: We have not yet inverted the mask, therefore making
                        # dilation at this stage is equivalent to erosion and vice versa
                        # (we dilate the white pixels that are those of the wall).
                        # ----------------------------
                        print_next_step(generator, "Dilate and Erode:")
                        kernel_size = 20  # 18 or 20
                        start_time = time.time()
                        dilated_wall_mask = image_dilation(wall_mask, kernel_size)
                        exe_time_dilation = time.time() - start_time
                        total_time += exe_time_dilation

                        show_image('image_dilation', dilated_wall_mask, height=405, width=720)

                        start_time = time.time()
                        eroded_wall_mask = image_erosion(dilated_wall_mask, kernel_size)
                        exe_time_erosion = time.time() - start_time
                        total_time += exe_time_erosion

                        print("\ttime: {:.3f} s".format(exe_time_dilation + exe_time_dilation))
                        show_image('image_erosion', eroded_wall_mask, height=405, width=720)

                        # Step 4: Invert the wall mask
                        # ----------------------------
                        print_next_step(generator, "Invert Wall Mask:")
                        start_time = time.time()
                        wall_mask_inverted = invert_image(eroded_wall_mask)
                        exe_time_invertion = time.time() - start_time
                        total_time += exe_time_invertion

                        print("\ttime: {:.3f} s".format(exe_time_invertion))
                        show_image('image_inversion', wall_mask_inverted, height=405, width=720)
                        cv2.imwrite(
                            os.path.join(dst_dir, photo.split('.')[0] + "_inverted_mask." + photo.split('.')[1]),
                            wall_mask_inverted)

                print()
                print("-" * 30)
                print("Total execution time: {:.3f} s".format(total_time))
