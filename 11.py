
import cv2

img = cv2.resize(cv2.imread('4.pgm', 0), (0, 0), fx=1.0, fy=1.0)
template = cv2.resize(cv2.imread('sh2.jpg', 0), (0, 0), fx=1.0, fy=1.0)
h, w = template.shape

methods = [cv2.TM_CCOEFF]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    print(img.shape)
    print(template.shape)
    (533, 533, 3)
    (100, 100, 3)

    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, 255, 5)
    cv2.imshow('Match', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
