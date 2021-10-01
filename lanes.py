import cv2
import numpy as np

global_left_fit_average = []
global_right_fit_average = []


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1 * (1 / 2))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    global global_left_fit_average
    global global_right_fit_average
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if not left_fit:
        left_fit_average = global_left_fit_average
    else:
        left_fit_average = np.average(left_fit, axis=0)
        global_left_fit_average = left_fit_average

    right_fit_average = np.average(right_fit, axis=0)
    global_right_fit_average = right_fit_average
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 100)


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)


fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 60, (1280, 720))

cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(
        cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5
    )
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    out.write(final_image)
    cv2.imshow("output", final_image)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
