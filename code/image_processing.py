# Testing MODE ON
import cv2
import numpy as np
import os
import time


def mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level):
    """Groups pixels together by colour and location.

    This function takes an image and mean-shift parameters and returns a version
    of the image that has had mean shift segmentation performed on it.

    Mean shift segmentation clusters nearby pixels with similar pixel values
    sets them all to have the value of the local maxima of pixel value.

    For details visit:
    - https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#pyrmeanshiftfiltering
    - https://docs.opencv.org/master/d7/d00/tutorial_meanshift.html

    Parameters
    ----------
    img: ndarray
        image to apply the Mean Shift Segmentation
    spatial_radius: int
        The spatial window radius
    color_radius: int
        The color window radius
    maximum_pyramid_level: int
        Maximum level of the pyramid for the segmentation

    Returns
    -------
    ndarray
        filtered “posterized” image with color gradients and fine-grain
        texture flattened
    """

    dst_img = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius, maximum_pyramid_level)
    return dst_img


def get_mask_largest_segment(img, color_difference=1, x_samples=8, skip_white=False):
    """Create a mask using the largest segment (this segment will be white).

    This is done by setting every pixel that is not the same color of the wall
    to have a value of 0 and every pixel has a value within a euclidean distance
    of `color_difference` to the wall's pixel value to have a value of 255.

    Parameters
    ----------
    img: ndarray
        image to apply masking
    color_difference: int
        euclidean distance between wall's pixel and the rest of the image
    x_samples : int
        numer of samples that will be tested orizontally in the image

    Returns
    -------
    ndarray
        Returns a version of the image where the wall is white and the rest of
        the image is black.
    """

    h, w, chn = img.shape
    color_difference = (color_difference,) * 3

    # in that way for smaller images the stride will be lower
    stride = int(w / x_samples)

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    wallmask = mask
    largest_segment = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if mask[y + 1, x + 1] == 0 or skip_white:
                mask[:] = 0
                # Fills a connected component with the given color.
                # For details visit:
                # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill#floodfill
                rect = cv2.floodFill(
                    image=img.copy(),
                    mask=mask,
                    seedPoint=(x, y),
                    newVal=0,
                    loDiff=color_difference,
                    upDiff=color_difference,
                    flags=4 | (255 << 8),
                )

                # For details visit:
                # https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57

                # Next operation is not necessary if flag is equal to `4 | ( 255 << 8 )`
                # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                segment_size = mask.sum()
                if segment_size > largest_segment:
                    largest_segment = segment_size
                    wallmask = mask[1:-1, 1:-1].copy()
                    # cv2.imshow('rect[2]', mask)
                    # cv2.waitKey(0)
    return wallmask


def mask_dilation(mask, kernel_size, kernel_value=255):
    """Dilate the image to remove noise.

    Dilation involves moving a kernel over the pixels of a binary image. When
    the kernel is centered on a pixel with a value of 0 and some of its pixels
    are on pixels with a value of 255, the centre pixel is given a value of 255.

    Parameters
    ----------
    mask: ndarray
        the mask to dilate
    kernel_size: int
        the kernel size
    kernel_value: int
        the kernel value (the value of each kernel pixel)

    Returns
    -------
    ndarray
        Returns the dilated mask
    """

    kernel = np.ones((kernel_size, kernel_size)) * kernel_value
    dilated_mask = cv2.dilate(wallmask, kernel)
    return dilated_mask


if __name__ == '__main__':
    photos_path = 'dataset/photos'
    videos_dir_name = '013'
    filename = '20180529_112417_ok_0031.jpg'

    total_time = 0

    img_path = os.path.join(photos_path, videos_dir_name, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(type(img))
    print(img.shape)
    cv2.imshow('image_original', img)

    # Step 1: Perform mean shift segmentation on the image
    start_time = time.time()
    spatial_radius = 7
    color_radius = 13
    maximum_pyramid_level = 1
    img_mss = mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level)
    exe_time_mean_shift_segmentation = time.time() - start_time
    print("Mean Shift Segmentation:\n\ttime: {:.3f} s".format(exe_time_mean_shift_segmentation))
    cv2.imshow('image_mean_shift_segmentation', img_mss)

    # Step 2: Get a mask of just the wall in the gallery
    start_time = time.time()
    color_difference = 2
    x_samples = 8
    wallmask = get_mask_largest_segment(img_mss, color_difference, x_samples)
    exe_time_mask_largest_segment = time.time() - start_time
    print("Mask Largest Segment\n\ttime: {:.3f} s".format(exe_time_mask_largest_segment))
    cv2.imshow('image_mask_largest_segment', wallmask)

    # Step 3: Dilate the wall mask to remove noise
    start_time = time.time()
    kernel_size = 18
    dilated_wall_mask = mask_dilation(wallmask, kernel_size, 255)
    exe_time_dilation = time.time() - start_time
    print("Dilation\n\ttime: {:.3f} s".format(exe_time_dilation))
    cv2.imshow('image_dilation', dilated_wall_mask)

    # Step 4: Invert the wall mask
    # Invert the wall mask for finding possible painting components
    start_time = time.time()
    inverted_wall_mask = cv2.bitwise_not(dilated_wall_mask)
    exe_time_invertion = time.time() - start_time
    print("Inversion\n\ttime: {:.3f} s".format(exe_time_invertion))
    cv2.imshow('image_inversion', inverted_wall_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
