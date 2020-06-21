from yolo.people_detection import PeopleDetection
import cv2



if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread('data_test\persone.jpg')
    p_detection = PeopleDetection()
    ris, _ = p_detection.run(img)
    cv2.imshow("Ris", ris)
    cv2.waitKey()