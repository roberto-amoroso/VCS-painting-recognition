from utils import check_media_file
from model.media_type import MediaType
import numpy as np
import cv2
import os
import sys
import enum

from draw import print_nicer

if __name__ == '__main__':
    base_path = 'dataset/videos'
    videos_dir_name = '000'
    # filename = "VID_20180529_112627_0000.jpg"
    filename = "VIRB0391.MP4"

    filename = os.path.join(base_path, videos_dir_name, filename)

    # ---------------------------------
    # Check if the input path is an image or a file
    # ---------------------------------

    media, media_type = check_media_file(filename)

    print(f"\t-Filename: {filename}")
    print(f"\t-Media_type: ", end='')
    if media_type == MediaType.image:
        print(f"{MediaType.image.name}")
    else:
        print(f"{MediaType.video.name}")
