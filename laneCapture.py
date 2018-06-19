# Code source from Udacity Lesson "Advance Techniques for Lane Finding"
import numpy as np
import cv2
import matplotlib.pyplot as plt

MARGIN = 100
N_WINDOWS = 9
MIN_PIXEL = 50

XM_PER_PIX = 3.7/700 # meters per pixel in y dimension
YM_PER_PIX = 30/720 # meters per pixel in x dimension

def sliding_win_fit(warped, nwindows=N_WINDOWS, margin=MARGIN,
                     minpix=MIN_PIXEL, line=None):
    """ Given a bird-eye view of the road, polynomials fits the 
    left and right lanes, and return these two polynomial fits
    
    Arguments:
        warped {np.array} -- a binarized image after perspective
            transform.
    
    Keyword Arguments:
        nwindows {int} -- number of sliding windows 
            (default: {9})
        margin {int} -- width of the windows +/- margin 
            (default: {100})
        minpix {int} -- minimum number of pixels found to recenter window 
            (default: {50})

    Return:
        left_fit {array} -- coefficients for a 2nd degree polynomial fit
        right_fit {array} -- coefficients for a 2nd degree polynomial fit
        out_img {np.array} -- image with windows drawn
    """
    # Set height of windows
    window_height = np.int(warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped, warped, warped))*255

    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    # Ignore the leftmost and rightmost quarter of the image
    # to reduce noise
    # TODO: Maybe there is a more robust method to replace this.
    quarter_point = np.int(midpoint//2)
    # Only the quarter directly to the left/right of the midpoint is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), 
            (win_xleft_high,win_y_high), (0,255,0), 3) 
        cv2.rectangle(out_img, (win_xright_low,win_y_low), 
            (win_xright_high,win_y_high), (0,255,0), 3) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & 
                          (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & 
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & 
                           (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & 
                           (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, out_img

# def fit_from_prev_frame(warped, line ):
#     nonzero = warped.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     margin = 100
#     left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
#     left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
#     left_fit[1]*nonzeroy + left_fit[2] + margin))) 

#     right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
#     right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
#     right_fit[1]*nonzeroy + right_fit[2] + margin)))  

#     # Again, extract left and right line pixel positions
#     leftx = nonzerox[left_lane_inds]
#     lefty = nonzeroy[left_lane_inds] 
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
#     # Fit a second order polynomial to each
#     left_fit = np.polyfit(lefty, leftx, 2)
#     right_fit = np.polyfit(righty, rightx, 2)

def get_dist_from_center(warped, left_fit, right_fit, xm_per_pix=XM_PER_PIX):
    """Calculate the distance from the center of the lane
    
    Arguments:
        warped {np.array} -- a binarized image after perspective
            transform.
        left_fit {array} -- coefficients for a 2nd degree polynomial fit
        right_fit {array} -- coefficients for a 2nd degree polynomial fit
    
    Keyword Arguments:
        xm_per_pix {float} -- pixel to meter ratio (default: {3.7/700})
    
    Returns:
        [float] -- distance away from the center of the lane in UNIT METERS.
    """
    # The bottom-most of the image
    max_y = warped.shape[0]-1
    # Calculate Distance from center of the lane
    car_position = warped.shape[1] / 2
    center_fit = (left_fit + right_fit) / 2
    # lane_center = (left_fitx + right_fitx) / 2
    lane_offset = center_fit[0] * max_y ** 2 + center_fit[1] * max_y + center_fit[2]
    dist = (car_position - lane_offset) * xm_per_pix
    # print("Distance from Center of Lane {:.3f} m".format(dist)) 
    return dist

def get_curvature(warped, left_fit, right_fit, 
                    xm_per_pix=XM_PER_PIX, ym_per_pix=YM_PER_PIX):
    # Define y-value where we want radius of curvature
    y_eval = warped.shape[0]-1
    # New polynomial coefficients after scaling
    # BEFORE: x =        a*(y**2)         +     b*y     + c
    # AFTER:  x = mx/(my ** 2)*a*(y ** 2) + (mx/my)*b*y + c
    left_fit_cr = [xm_per_pix / (ym_per_pix ** 2) * left_fit[0],
                   (xm_per_pix / ym_per_pix) * left_fit[1],
                   left_fit[2]]
    right_fit_cr = [xm_per_pix / (ym_per_pix ** 2) * right_fit[0],
                   (xm_per_pix / ym_per_pix) * right_fit[1],
                   right_fit[2]]
    # Calculate the new radii of curvature
    left_curverad =  ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print("Left Lane Curvature: {:.3f} m, Right Lane Curvature: {:.3f} m".format(left_curverad, right_curverad))
    return left_curverad, right_curverad

def draw_lane(orig, warped, Minv, left_fit, right_fit):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orig.shape[1], orig.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(orig, 1, newwarp, 0.3, 0)

    return result