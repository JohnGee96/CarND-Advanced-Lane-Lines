from imgfilter import binarize
from calibrate import *
from laneCapture import *
from line import Line
from cv2 import putText, FONT_HERSHEY_DUPLEX

COEF_PATH = "./presets/calibrate.p"

dist_pickle = pickle.load(open(COEF_PATH, "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def write_info_on_img(img, left_curv, right_curv, dist_from_center):
    font = cv2.FONT_HERSHEY_DUPLEX
    curvature = 0.5 * (left_curv + right_curv) / 1000
    reportTxt1 = "Dist from Center: {:.2f} m".format(dist_from_center)
    reportTxt2 = "Radius of Curvature: {:.1f} km".format(curvature)
    cv2.putText(img, reportTxt1, (40,40), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, reportTxt2, (40,80), font, 1, (255,255,255), 2, cv2.LINE_AA) 
    return img

def find_lane(img, mtx, dist, line):
    undist = undistort(img, mtx, dist)
    binary, _ = binarize(undist)
    warped, _, Minv = bird_eye_view(binary)
    left_fit, right_fit, _ = sliding_win_fit(warped)
    result = draw_lane(img, warped, Minv, left_fit, right_fit)
    dist_from_center = get_dist_from_center(warped, left_fit, right_fit)
    left_curv, right_curv = get_curvature(warped, left_fit, right_fit)
   
    return write_info_on_img(result, left_curv, right_curv, dist_from_center)