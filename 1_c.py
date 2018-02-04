import numpy as np # use numpy library as np for array object
import cv2 # opencv-python

# segment the pool table used color information
def table_selection(image):
    boundaries = [ ([60, 60, 0], [190, 150, 35])] # color boundary for the pool table
    
    # create NumPy arrays from the boundary
    for (lower, upper) in boundaries:
    	lower = np.array(lower, dtype = "uint8")
    	upper = np.array(upper, dtype = "uint8")
     
    	ts_mask = cv2.inRange(image, lower, upper) # create the mask for the selected region
    	result = cv2.bitwise_and(image, image, mask = ts_mask) # remain only the pool table
        
    return result, ts_mask # return table selection and mask


# select the largest pool table and remove the green region
def largest_pool_table(ori_image, table_img_mask):
    
    ret,thresh = cv2.threshold(table_img_mask, 40, 255, 0) # convert the mask image into binary
    
    ## find contours
    im,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0: # if has contour
        c = max(contours, key = cv2.contourArea) # find the largest contour area(select the largest pool table)

    ## create a table mask only for the green region
    mask = np.zeros(ori_image.shape, dtype='uint8') # create a empty object for the table mask (green region) 
    mask = cv2.drawContours(mask, [c], -1, (225,225,225), -1) # create a mask for the pool table (green region) 
    fat_mask = cv2.dilate(mask, None, iterations=10) # expand the mask
    ret,thresh = cv2.threshold(fat_mask,127,255,cv2.THRESH_BINARY_INV) # transfer into inverse binary mask
    gray_mask = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY) # convert the mask into grayscale
    
    ## create a table mask
    mask_for_table = np.zeros(ori_image.shape, dtype='uint8') # generate a empty object for the table mask
    mask_point = np.array([[[850,513], [30,880], [1340,1130], [1450,540]]], dtype = np.int32) #select mask point
    cv2.fillPoly(mask_for_table, mask_point, (255, 255, 255)) # generate pool table mask
    gray_mask_4_table = cv2.cvtColor(mask_for_table, cv2.COLOR_BGR2GRAY) # convert the mask image into grayscale
     
    ## select the edge of the pool table by useing combined mask
    comb_mask = cv2.bitwise_and(gray_mask_4_table, gray_mask_4_table, mask=gray_mask) #create the mask only for table edges
    edg_img = cv2.bitwise_and(ori_image, ori_image, mask=comb_mask) # select the edges of the table
    return edg_img # return the edges image for corner detection

# find the corner by using Harris Corner Detector
def Harris_corner(ori_image, edg_img):
    gray = cv2.cvtColor(edg_img,cv2.COLOR_BGR2GRAY) # convert the edge image into grayscale
    
    # find Harris corners
    gray = np.float32(gray) # change data type to float 32 
                            # (src: Input single-channel 8-bit or floating-point image)
    dst = cv2.cornerHarris(gray,2,3,0.04) # using opencv function "cv2.cornerHarris"
    dst = cv2.dilate(dst,None) # image dilate for marking the corners
    ret, dst = cv2.threshold(dst,0.005*dst.max(),255,0) # threshold for optimal value
    
    dst = np.uint8(dst) # change data type to unit8
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # Corner with SubPixel Accuracy
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    # get the corner point position
    res = np.hstack((centroids,corners))
    res = np.int0(res)

    a, b = res.shape # get the number of corner points 
    img_row, img_col, img_dep = ori_image.shape # get the input image shape

    point_selection = [] # select 4 extreme point for the table corner
    second = [] # use to adjust the point selection
    
    # find the top-right corner point
    temp = 0
    for m in range (a):
        if temp < res[m,2]: # compare every points in the res to find the top-right point 
            temp = res[m,2]
            temp_num = m
    point_selection.append(temp_num) # append the result into point_selection
    
    # find the down-left corner point
    temp = img_col
    for m in range (a):
        if temp > res[m,2]: # compare every points in the res to find the down-left point
            temp = res[m,2]
            temp_num = m
    point_selection.append(temp_num) # append the result into point_selection
    
    # find the down-right corner point
    temp = 0        
    for m in range (a):
        if temp < res[m,3]: # compare every points in the res to find the down-right point
            temp = res[m,3]
            temp_num = m
            second.append(temp_num) # adjust the result point
    
    point_selection.append(second[len(second)-2]) # append the result into point_selection
    
    # find the top-left corner point        
    temp = img_row        
    for m in range (a):
        if temp > res[m,3]: # compare every points in the res to find the top-left point
            temp = res[m,3]
            temp_num = m
    point_selection.append(temp_num) # append the result into point_selection

    # draw the four courner points which been selected
    for i in range(4):
        cv2.circle(ori_image,(res[point_selection[i],2],res[point_selection[i],3]), 25, (0,0,255), -1)
        
    return ori_image #return result image
#-------------------------------------------------------#
ori_image = cv2.imread('pool.jpg')                      # read image
table_img, table_img_mask = table_selection(ori_image)  # segment the pool table used color information
seg_obj = largest_pool_table(ori_image, table_img_mask) # slect the largest pool table
corner_point = Harris_corner(ori_image, seg_obj)        # find the corner for the pool table
#-------------------------------------------------------#   


#cv2.imwrite("1_c.jpg", corner_point) # output table mask for problem 1, part b
cv2.imshow("1_c", corner_point) # show result image
cv2.waitKey(0) # system pause






