import cv2
import math as m

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye_tree_eyeglasses.xml")


img = cv2.imread("IMG_4531.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)
for (x, y, w, h) in faces:
    center_x = int(x + 0.5 * w)
    center_y = int(y + 0.5 * h)
    cv2.line(img, (center_x, center_y + 1000), (center_x, center_y - 1000), (0, 255, 0), 5)

    eyes = eye_cascade_db.detectMultiScale(img_gray)

    for (ex, ey, ew, eh) in eyes:
        eye_x = int(ex + 0.5 * ew)
        eye_y = int(ey + 0.5 * eh)
        cv2.line(img, (eye_x, eye_y + 1000), (eye_x, eye_y - 1000), (0, 255, 0), 5)
        distance = m.sqrt((center_x - eye_x) ** 2 + (center_y - eye_y) ** 2)
        print(distance)

    cv2.imshow('rez', img)
    cv2.waitKey()
