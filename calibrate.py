import pickle
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NY = 6 # Num of corners in the y direction
NX = 9 # Num of corners in the x direction

CAL_IMG_PATH = 'camera_cal/calibration*.jpg'
SAVE_PATH = './presets/calibrate.p'

OFFSET = 260
# Corner coordinates picked from "./test_images/straigt_lines1.jpg"
CORNERS = np.float32([[190,720],[600,447],[685,447],[1125,720]])
# CORNERS = np.float32([[190,720],[585,455],[700,455],[1130,720]])

def calibrate_camera(save_file=False, cal_img_path=CAL_IMG_PATH, save_path=SAVE_PATH):
    """ 
    Create and save distortion coefficients for calibrating the camera
    Implementation models after the code from the Lesson 15 
    in section `Calibrating Your Camera` of Udacity's CarND 
    """
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image space

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((NY * NX, 3), np.float32)
    objp[:,:2] = np.mgrid[0:NX, 0:NY].T.reshape(-1,2)

    images = glob.glob(cal_img_path)

    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Do camera calibration given object points and image points
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)

    if save_file:
        # Save the camera calibration
        print("Save result to", save_path)
        result_pickle = {}
        result_pickle["mtx"] = mtx
        result_pickle["dist"] = dist
        pickle.dump(result_pickle, open(save_path, "wb" ))

    return mtx, dist

def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def bird_eye_view(img, offset=OFFSET, corners=CORNERS):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(corners)
                      # from Bot-left, clockwise
    dst = np.float32([[corners[0][0] + offset, corners[0][1]],
                      [corners[0][0] + offset, 0],
                      [corners[3][0] - offset, 0],
                      [corners[3][0] - offset, corners[3][1]]])   
                      
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR)    
    return warped, M, Minv