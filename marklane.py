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
    height_middle = int(sx.shape[0]/2)
    mag = sobel_magnitude(sx, sy, wx, wy)
    mag = mag/np.max(mag[height_middle:,:]) # only consider lower half
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
    s_channel = hls[:,:,2]
    sobel_x = abs_sobel(gray, orient='x')
    sobel_y = abs_sobel(gray, orient='y')
    sobel_rel_mag = sobel_magnitude_thresh(\
            sobel_x, sobel_y, wy=0.1, thresh=(0.02,1))
    # denoise
    sobel_rel_mag = cv2.GaussianBlur(sobel_rel_mag, (13,13), 0)
    sobel_rel_mag = (sobel_rel_mag>0.7)
    # sobel highlights boundaries, use filter2D to expand to neighboring pixels
    sobel_rel_mag = np.clip(cv2.filter2D(sobel_rel_mag.astype(np.float64),\
            -1,np.ones((9,9))),0,1)

    # filter out certain directions
    sobel_dir = sobel_direction_thresh(sobel_x, sobel_y, \
            thresh=(np.pi/2.5, np.pi/2))
    sobel_dir = cv2.GaussianBlur(1-sobel_dir, (13,13), 0)
    sobel_dir = (sobel_dir>0.7)

    s_channel = sobel_magnitude_thresh(\
            abs_sobel(s_channel, orient='x', sobel_kernel=5), \
            abs_sobel(s_channel, orient='y', sobel_kernel=5), \
            wy=0.1, thresh=(0.02,1))
    s_channel = cv2.GaussianBlur(s_channel, (13,13), 0)
    s_channel = (s_channel>0.7)
    s_channel = np.clip(cv2.filter2D(s_channel.astype(np.float64),\
            -1,np.ones((9,9))),0,1)
    res = cv2.GaussianBlur(s_channel*sobel_rel_mag*sobel_dir, (13,13), 0)
    res = 255*((res>0.8).astype(np.uint8))
    return res

def get_perspective_matrix():
    """ returns the perspective transform matrix and its inverse """
    pp_src = np.array([(200,684),(1120,684),(542,475),(743,475)]).astype(np.float32)
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
    window[:int(window_width/2)] *= -1
    # Define pixel count threshold for level estimate
    # Windows with count lower than this are excluded
    mask_thresh = window_height*5*255
    # Define Fractional height and threshold of the initial estimate
    init_frac = 1/4
    init_thresh = warped.shape[0]*init_frac / window_height * mask_thresh
    # Store the (left,right) window centroid positions per level
    # either entry can be None
    window_centroids = [] 
    # Initialize the centers for search
    l_center, r_center = None, None
    # keep track of the direction in which the lane is moving to help detection
    delta, delta_score = 0., 1.
    old_weight, new_weight = 0.7, 0.3
    # Go through each layer looking for max pixel locations
    for level in range(0,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[\
                int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        init_layer = np.sum(warped[\
                int(warped.shape[0]-level*window_height-warped.shape[0]*init_frac):int(warped.shape[0]-level*window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        init_conv_signal = np.convolve(window, init_layer)
        
        # try to find where the lane is going next
        level_delta, level_delta_score = 0., 0.
        if l_center:
            l_min_index = np.clip(l_center+offset-margin+delta, 0, warped.shape[1]).astype(np.int)
            l_max_index = np.clip(l_center+offset+margin+delta, 0, warped.shape[1]).astype(np.int)
            conv_ptr = conv_signal
        else:
            l_min_index = int(warped.shape[1]*0.15)
            l_max_index = int(warped.shape[1]*0.45)
            conv_ptr = init_conv_signal
        if l_min_index<l_max_index:
            l_center_new = np.argmax(conv_ptr[l_min_index:l_max_index])+l_min_index-offset
            l_center_new = l_center_new.astype(np.int)
            l_center_offset = min(l_center_new+offset, warped.shape[1]).astype(np.int)
            if conv_ptr[l_center_offset] < mask_thresh:
                l_center_new = None
            else:
                if l_center:
                    level_delta += (l_center_new - l_center) * ((conv_ptr[l_center_offset]/255)**2)
                    level_delta_score += (conv_ptr[l_center_offset]/255)**2
        if r_center:
            r_min_index = np.clip(r_center+offset-margin+delta, 0, warped.shape[1]).astype(np.int)
            r_max_index = np.clip(r_center+offset+margin+delta, 0, warped.shape[1]).astype(np.int)
            conv_ptr = conv_signal
        else:
            r_min_index = int(warped.shape[1]*0.55+offset)
            r_max_index = int(warped.shape[1]*0.85)
            conv_ptr = init_conv_signal
        if r_min_index<r_max_index:
            r_center_new = np.argmin(conv_ptr[r_min_index:r_max_index])+r_min_index-offset
            r_center_new = r_center_new.astype(np.int)
            r_center_offset = min(r_center_new+offset, warped.shape[1]).astype(np.int)
            if -conv_ptr[r_center_offset] < mask_thresh:
                r_center_new = None
            else:
                if r_center:
                    level_delta += (r_center_new - r_center) * ((-conv_ptr[r_center_offset]/255)**2)
                    level_delta_score += ((-conv_ptr[r_center_offset]/255)**2)
        delta, delta_score = (delta*delta_score*old_weight + \
                              level_delta*new_weight)/(delta_score*old_weight+level_delta_score*new_weight), \
                             delta_score*old_weight+level_delta_score*new_weight
        
        # now check if windows at the new delta direction are good
        if l_center:
            l_center_offset = np.clip(l_center+delta+offset, 0, warped.shape[1]).astype(np.int)
            if conv_signal[l_center_offset] >= mask_thresh:
                l_center_new = int(l_center+delta)
            else:
                l_center_new = None
            l_center = int(l_center+delta)
        else:
            l_center = l_center_new
        
        if r_center:
            r_center_offset = np.clip(r_center+delta+offset, 0, warped.shape[1]).astype(np.int)
            if -conv_signal[r_center_offset] >= mask_thresh:
                r_center_new = int(r_center+delta)
            else:
                r_center_new = None
            r_center = int(r_center + delta)
        else:
            r_center = r_center_new
        window_centroids.append((l_center_new, r_center_new))

    return window_centroids

def pixel_in_windows(img, window_centroids, window_width, window_height, \
        sample = 20):
    """ 
    img has value either 0 or 255 
    returns array of points that are identified as lanes, sampled by level

    no longer needed in the latest implementation
    """
    offset = window_width / 2
    level = img.shape[0]
    resultx, resulty = None, None
    for c in window_centroids:
        if c:
            y_min = int(c-offset)
            y_max = int(c+offset)
            x_min = level-window_height
            x_max = level
            retx, rety = np.where(img[x_min:x_max,y_min:y_max]==255)
            if retx.shape[0]>0:
                prob = min(sample / retx.shape[0], 1)
                # prob = sample_ratio
                sample_prob = (np.random.random(retx.shape[0])<=prob)
                retx, rety = retx[sample_prob], rety[sample_prob]
                if retx.shape[0] > 0:
                    retx += x_min
                    rety += y_min
                    if resultx is not None:
                        resultx = np.hstack([resultx, retx])
                        resulty = np.hstack([resulty, rety])
                    else:
                        resultx, resulty = retx, rety
        level -= window_height
    return resultx, resulty

def fit_lane_centroids(\
        warped, \
        window_centroids, \
        window_width, \
        window_height):
    """
    warped - binary warped image
    window_centroids - windows containing lane pixels
    window_width and window_height - window dimensions

    returns a list of arrays containing the coordinates of the
    windows (found and predicted) that can be used to fit polynomials
    and plot windows
    """

    # first estimate lane width in terms of pixels
    lx, ly, rx, ry = [], [], [], [] # centroids that are detected
    flx, fly, frx, fry = [], [], [], [] # centroids that are fitted
    diffs = []
    for level in range(0,(int)(warped.shape[0]/window_height)):
        if window_centroids[level][0] and window_centroids[level][1]:
            diffs.append(window_centroids[level][0]-window_centroids[level][1])
    diff_mean = None
    if diffs:
        diff_mean = np.mean(diffs)
    # now add the windows as well as the predicted windows
    for level in range(0,(int)(warped.shape[0]/window_height)):
        x_center = warped.shape[0]-(level+0.5)*window_height
        if window_centroids[level][0]:
            lx.append(x_center)
            ly.append(window_centroids[level][0])
            flx.append(x_center)
            fly.append(window_centroids[level][0])
        elif window_centroids[level][1] and diff_mean:
            flx.append(x_center)
            fly.append(window_centroids[level][1]+diff_mean)
        if window_centroids[level][1]:
            rx.append(x_center)
            ry.append(window_centroids[level][1])
            frx.append(x_center)
            fry.append(window_centroids[level][1])
        elif window_centroids[level][0] and diff_mean:
            frx.append(x_center)
            fry.append(window_centroids[level][0]-diff_mean)
    # fit polynomial
    lx, ly = np.array(lx), np.array(ly)
    rx, ry = np.array(rx), np.array(ry)
    flx, fly = np.array(flx), np.array(fly)
    frx, fry = np.array(frx), np.array(fry)
    fly -= int(window_width/4)
    fry += int(window_width/4)
    return lx, ly, rx, ry, flx, fly, frx, fry

def draw_lanes(img_ud, flx, fly, frx, fry, pp_mtx_inv, annotate=True):
    """
    img_ud - undistorted image on which lanes will be marked
    pp_mtx_inv - the inverse perspective transform matrix
    annotate - whether to put curvature numbers on image

    returns the image and the statistics
    if the input centroids are not sufficient, returns the original
    image and None for statistics
    """
    if flx.shape[0]==0 or frx.shape[0]==0:
        return img_ud, None, None, None, None
    pl = np.polyfit(flx, fly, 2)
    pr = np.polyfit(frx, fry, 2)
    pts_y = np.arange(0, img_ud.shape[0], 1)
    pts_lx = pl[0]*pts_y**2 + pl[1]*pts_y + pl[2]
    pts_rx = pr[0]*pts_y**2 + pr[1]*pts_y + pr[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([pts_lx, pts_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([pts_rx, pts_y])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto a newly created warped blank image
    warp_zero = np.zeros_like(img_ud[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.fillPoly(color_warp, np.int_([pts.astype(np.int32)]), (0, 255, 0))
    # Warp the blank back to original image space using 
    # inverse perspective matrix 
    newwarp = cv2.warpPerspective(color_warp, pp_mtx_inv, \
            (img_ud.shape[1], img_ud.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img_ud, 1, newwarp, 0.3, 0)
    # now compute distance and curvature
    xm_per_pix = 30/720 # meters per pixel vertically
    ym_per_pix = 3.7/700 # meters per pixel horizontally
    x_eval = 720
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(flx*xm_per_pix, fly*ym_per_pix, 2)
    right_fit_cr = np.polyfit(frx*xm_per_pix, fry*ym_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*x_eval*xm_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*x_eval*xm_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # in theory averaging not necessary because the current window
    # detection algorithm ensures that the detected windows are parallel
    avg_curverad = 2/(1/left_curverad+1/right_curverad) 
    lane_center = (np.polyval(left_fit_cr,x_eval*xm_per_pix)+\
            np.polyval(right_fit_cr,x_eval*xm_per_pix))/2
    lane_loc = ym_per_pix*1280/2-lane_center
    if annotate:
        cv2.putText(result, \
                '{0} curve avg {1:.3f}m'.format(\
                'left' if left_fit_cr[0]+right_fit_cr[0]<0 else 'right', \
                avg_curverad), \
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
        window_height, \
        lx, ly, rx, ry, flx, fly, frx, fry):
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
    if lx.shape[0]==0 or rx.shape[0]==0:
        return output
    pl = np.polyfit(flx, fly, 2)
    pr = np.polyfit(frx, fry, 2)
    linex = np.arange(0,720,10)
    linely = np.polyval(pl, linex)
    linery = np.polyval(pr, linex)
    cv2.polylines(output, \
            [np.vstack([linely,linex]).T.reshape(-1,1,2).astype(np.int32), \
            np.vstack([linery,linex]).T.reshape(-1,1,2).astype(np.int32)], \
            False, (0,0,255), 10)
    return output

