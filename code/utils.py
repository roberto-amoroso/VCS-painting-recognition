import cv2
import numpy as np


def step_generator():
    start = 1
    while True:
        yield start
        start += 1


def print_next_step(generator, title):
    step = next(generator)
    print(f"\n# Step {step}: {title}")
    # print("-" * 30)
    pass


def show_image(title, img, height=405, width=720):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width, height)
    cv2.imshow(title, img)


def draw_lines(img, lines, probabilistic_mode=True):
    # Draw the lines:
    # copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if probabilistic_mode:
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    else:
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Detected Lines (in red) - {} Hough Line Transform".format(
        "Probabilistic" if probabilistic_mode else "Standard"), cdst)


if __name__ == '__main__':
    # for num in step_generator("A"):
    #     if num > 5:
    #         break
    #     print(num)

    generator = step_generator()

    print_next_step(generator, "A")
    print_next_step(generator, "B")

    # dilated_wall_mask = image_dilation(wall_mask_inverted, 50)
    # show_image('test1', dilated_wall_mask)
    #
    # eroded_wall_mask = image_erosion(dilated_wall_mask, 60)
    # show_image('test2', eroded_wall_mask)
    #
    # eroded_wall_mask = image_morphology_tranformation(dilated_wall_mask, cv2.MORPH_OPEN, 60)
    # show_image('test3', eroded_wall_mask)
