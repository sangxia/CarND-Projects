import numpy as np
import cv2

def undistort(img, dist_info):
    """ 
    returns an undistorted image
    dist_info is a dictionary with fields mtx and dist
    """
    return cv2.undistort(\
            img, \
            dist_info['mtx'], \
            dist_info['dist'], \
            None, \
            dist_info['mtx'])

def abs_sobel(img, orient='x', sobel_kernel=9):
    """ img should already be grayscale """
    ret = np.absolute(cv2.Sobel(\
            img, \
            cv2.CV_64F, \
            1*(orient=='x'), \
            1*(orient=='y'), \
            ksize=sobel_kernel))
    return ret

def sobel_magnitude(sx, sy, wx=1, wy=1):
    """ sx and sy are sobel magnitures, wx and wy are weights """
    return np.sqrt(wx*sx**2+wy*sy**2)

def sobel_magnitude_thresh(sx, sy, wx=1, wy=1, thresh=(0.3,1)):
    mag = sobel_magnitude(sx, sy, wx, wy)
    mag = mag/np.max(mag)
    ret = np.zeros_like(mag)
    ret[(mag>=thresh[0]) & (mag<=thresh[1])] = 1
    return ret

def sobel_direction_thresh(sx, sy, thresh=(np.pi/6, np.pi/6*2.5)):
    dr = np.arctan2(sy, sx)
    ret = np.zeros_like(dr)
    ret[(dr>=thresh[0]) & (dr<=thresh[1])] = 1
    return ret

def binary_lane_threshold(img_ud):
    """ 
    img_ud is an undistorted image in BGR 
    output array contains value either 0 or 255
    """
    img_size = (img_ud.shape[1], img_ud.shape[0])
    hls = cv2.cvtColor(img_ud, cv2.COLOR_BGR2HLS)
    gray = img_ud[:,:,-1] # use red channel only
    sobel_x = abs_sobel(gray, orient='x')
    sobel_y = abs_sobel(gray, orient='y')
    sobel_rel_mag = sobel_magnitude_thresh(\
            sobel_x, sobel_y, wy=0, thresh=(0.05,1))
    # sobel highlights boundaries, use filter2D to expand to neighboring pixels
    sobel_rel_mag = np.clip(cv2.filter2D(sobel_rel_mag,-1,np.ones((7,7))),0,1)
    s_channel = hls[:,:,2]
    s_channel = (s_channel > 160)
    res = 255*(s_channel*sobel_rel_mag).astype(np.uint8)
    return res

def get_perspective_matrix():
    """ returns the perspective transform matrix and its inverse """
    pp_src = np.array([(200,720-36),(1120,720-36),(580,720-268),(705,720-268)]).astype(np.float32)
    pp_dst = np.array([(320,720),(960,720),(320,0),(960,0)]).astype(np.float32)
    pp_mtx = cv2.getPerspectiveTransform(pp_src, pp_dst)
    pp_mtx_inv = cv2.getPerspectiveTransform(pp_dst, pp_src)
    return pp_mtx, pp_mtx_inv

def warp_img(img, pp_mtx):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, pp_mtx, img_size)

def find_window_centroids(warped, window_width, window_height, margin):
    # offset from window center, because convolve aligns right
    offset = window_width/2 
    # Create our window template that we will use for convolutions
    window = np.ones(window_width) 
    # Define pixel count threshold for level estimate
    # Windows with count lower than this are excluded
    mask_thresh = window_height*0.3*255
    # Define Fractional height and threshold of the initial estimate
    init_frac = 1/4
    init_thresh = warped.shape[0]*init_frac / window_height * mask_thresh
    # Store the (left,right) window centroid positions per level
    # either entry can be None
    window_centroids = [] 
    # Initialize the centers for search
    l_center, r_center = None, None
    # Go through each layer looking for max pixel locations
    for level in range(0,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[\
                int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        init_layer = np.sum(warped[\
                int(warped.shape[0]-level*window_height-warped.shape[0]*init_frac):int(warped.shape[0]-level*window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        init_conv_signal = np.convolve(window, init_layer)
        
        if l_center:
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            conv_ptr = conv_signal
        else:
            l_min_index = int(warped.shape[1]/6)
            l_max_index = int(warped.shape[1]/2)
            conv_ptr = init_conv_signal
        l_center_new = np.argmax(conv_ptr[l_min_index:l_max_index])+l_min_index-offset
        l_center_new = l_center_new.astype(np.int)
        l_center_offset = min(l_center_new+offset, warped.shape[1]).astype(np.int)
        if conv_ptr[l_center_offset] > mask_thresh:
            l_center = l_center_new
        else:
            l_center_new = None
        
        if r_center:
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            conv_ptr = conv_signal
        else:
            r_min_index = int(warped.shape[1]/2+offset)
            r_max_index = int(warped.shape[1]*5/6)
            conv_ptr = init_conv_signal
        r_center_new = np.argmax(conv_ptr[r_min_index:r_max_index])+r_min_index-offset
        r_center_new = r_center_new.astype(np.int)
        r_center_offset = min(r_center_new+offset, warped.shape[1]).astype(np.int)
        if conv_ptr[r_center_offset] > mask_thresh:
            r_center = r_center_new
        else:
            r_center_new = None
        
        window_centroids.append((l_center_new, r_center_new))

    return window_centroids

def pixel_in_windows(img, window_centroids, window_width, window_height):
    """ 
    img has value either 0 or 255 
    returns array of points that are identified as lanes
    """
    offset = window_width / 2
    level = img.shape[0]
    ret = np.zeros_like(img)
    for c in window_centroids:
        if c:
            y_min = int(c-offset)
            y_max = int(c+offset)
            ret[level-window_height:level,y_min:y_max] = img[level-window_height:level,y_min:y_max]
        level -= window_height
    return np.where(ret == 255)

def draw_lanes(\
        img_ud, \
        warped, \
        window_centroids, \
        window_width, \
        window_height, \
        pp_mtx_inv, \
        annotate=True):
    """
    img_ud - undistorted image on which lanes will be marked
    warped - binary warped image
    window_centroids - windows containing lane pixels
    window_width and window_height - window dimensions
    pp_mtx_inv - the inverse perspective transform matrix
    annotate - whether to put curvature numbers on image

    returns the images with the lane marked and annotated if needed, 
    and the left, right and averaged curvature, as well as the relative 
    distance to lane center (negative if left to center, pos if right)

    if either lane is not found, returns None
    """
    # fit the polynomial for left and right lanes
    lx, ly = pixel_in_windows(warped, \
            [u for u,v in window_centroids], window_width, window_height)
    rx, ry = pixel_in_windows(warped, \
            [v for u,v in window_centroids], window_width, window_height)
    if lx.shape[0]==0 or rx.shape[0]==0:
        return None
    pl = np.polyfit(lx, ly, 2)
    pr = np.polyfit(rx, ry, 2)
    pts_y = np.arange(0, warped.shape[0], 1)
    pts_lx = pl[0]*pts_y**2 + pl[1]*pts_y + pl[2]
    pts_rx = pr[0]*pts_y**2 + pr[1]*pts_y + pr[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([pts_lx, pts_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([pts_rx, pts_y])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto a newly created warped blank image
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using 
    #inverse perspective matrix 
    newwarp = cv2.warpPerspective(color_warp, pp_mtx_inv, \
            (img_ud.shape[1], img_ud.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img_ud, 1, newwarp, 0.3, 0)
    # now compute distance and curvature
    xm_per_pix = 30/720 # meters per pixel vertically
    ym_per_pix = 3.7/700 # meters per pixel horizontally
    x_eval = 720
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lx*xm_per_pix, ly*ym_per_pix, 2)
    right_fit_cr = np.polyfit(rx*xm_per_pix, ry*ym_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*x_eval*xm_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*x_eval*xm_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    avg_curverad = 2/(1/left_curverad+1/right_curverad)
    lane_center = (np.polyval(left_fit_cr,x_eval*xm_per_pix)+\
            np.polyval(right_fit_cr,x_eval*xm_per_pix))/2
    lane_loc = ym_per_pix*1280/2-lane_center
    if annotate:
        cv2.putText(result, \
                '{0} curve avg {1:.3f}m (left {2:.3f}m, right {3:.3f}m)'.format(\
                'left' if left_fit_cr[0]+right_fit_cr[0]<0 else 'right', \
                avg_curverad, left_curverad, right_curverad), \
                (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        cv2.putText(result, \
                '{0:.3f}m {1}'.format(\
                abs(lane_loc), 'left' if lane_loc<0 else 'right'), \
                (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    return result, left_curverad, right_curverad, avg_curverad, \
            ym_per_pix*1280/2-lane_center

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),\
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def fit_warped(\
        warped, \
        window_centroids, \
        window_width, \
        window_height):
    result = np.zeros_like(warped)
    for level in range(0,len(window_centroids)):
        if window_centroids[level][0]:
            result += window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        if window_centroids[level][1]:
            result += window_mask(window_width,window_height,warped,window_centroids[level][1],level)
    result[result>0] = 255
    result = result.astype(np.uint8)
    zero_channel = np.zeros_like(result)
    # make window pixels green
    result = np.array(cv2.merge((zero_channel,result,zero_channel)),np.uint8) 
    # making the original road pixels 3 color channels
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) 
    output = cv2.addWeighted(warpage, 1, result, 0.5, 0.0) # overlay the orignal road image with window results
    # collect lane pixels and fit lines
    lx, ly = pixel_in_windows(warped, [u for u,v in window_centroids], window_width, window_height)
    rx, ry = pixel_in_windows(warped, [v for u,v in window_centroids], window_width, window_height)
    if lx.shape[0]==0 or rx.shape[0]==0:
        return output
    pl = np.polyfit(lx, ly, 2)
    pr = np.polyfit(rx, ry, 2)
    linex = np.arange(0,720,10)
    linely = np.polyval(pl, linex)
    linery = np.polyval(pr, linex)
    cv2.polylines(output, \
            [np.vstack([linely,linex]).T.reshape(-1,1,2).astype(np.int32), \
            np.vstack([linery,linex]).T.reshape(-1,1,2).astype(np.int32)], \
            False, (255,0,0), 10)
    return output

