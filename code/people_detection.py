from yolo.people_detection import PeopleDetection
import cv2
import os

if __name__ == "__main__":
    photos_path = 'dataset/photos'
    recognized_painting_path = 'dataset/recognized_paintings'
    videos_dir_name = '002'
    filename = "20180206_114604_0000.jpg"  # pople video '002'
    # filename = "yolo_test.jpg"  # web image for testing
    # filename = "VID_20180529_113001_0000.jpg"
    img_path = os.path.join(photos_path, videos_dir_name, filename)

    img = cv2.imread(img_path)

    cv2.imshow("Original", img)
    cv2.waitKey()

    p_detection = PeopleDetection()
    img_people_detected, people_in_frame, people_bounding_boxes = p_detection.run(img)

    print("There", end='')
    if people_in_frame:
        print(" are ", end='')
    else:
        print(" aren't ", end='')
    print("people in the frame.")

    cv2.imshow("Detected", img_people_detected)
    cv2.waitKey()
