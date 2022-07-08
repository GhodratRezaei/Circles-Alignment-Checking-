#######################################################      Packages  ##################################

# Import Package
import cv2
import numpy as np
import matplotlib.pyplot as plt
import maxflow


#######################################################      Functions   ##################################


# MRF Noise Removal
def postprocessing(im, unary):
	unary = np.float32(unary)
	unary = cv2.GaussianBlur(unary, (9, 9), 0)

	g = maxflow.Graph[float]()
	nodes = g.add_grid_nodes(unary.shape)

	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			v = nodes[i,j]
			g.add_tedge(v, 1-unary[i,j], unary[i,j])


	def potts_add_edge(i0, j0, i1, j1):
		v0, v1 = nodes[i0,j0], nodes[i1,j1]
		w = 0.1 * np.exp(-((im[i0,j0] - im[i1,j1])**2).sum() / 0.1)
		g.add_edge(v0, v1, w, w)


	for i in range(1,im.shape[0]-1):
		for j in range(1,im.shape[1]-1):
			potts_add_edge(i, j, i, j-1)
			potts_add_edge(i, j, i, j+1)
			potts_add_edge(i, j, i-1, j)
			potts_add_edge(i, j, i+1, j)

	g.maxflow()
	sgm = g.get_grid_segments(nodes)
	seg =np.float32(sgm)
	return seg




# Stacking Images
def stackImages(scale, imgArray):
    rows = len(imgArray)
    columns = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)  # will return  Boolean Value
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, columns):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:  imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), dtype='uint8')
        rows_list = [imageBlank] * rows
        for i in range(0, rows):
            rows_list[i] = imgArray[i]
        for i in range(0, rows):
            rows_list[i] = np.hstack(imgArray[i])
        ver = np.vstack(rows_list)

    else:

        for i in range(0, rows):
            if imgArray[i].shape[:2] == imgArray[0].shape[:2]:
                imgArray[i] = cv2.resize(imgArray[i], (0, 0), None, scale, scale)
            else:
                imgArray[i] = cv2.resize(imgArray[i], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[i].shape) == 2:
                imgArray[i] = cv2.cvtColor(imgArray[i], cv2.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)
        ver = hor

    return ver
#########################################################################################################################
img = cv2.imread('Beresha1.jpg')
img = img[100:500, 400:800]
img = cv2.resize(img, (800,800))

# GrayScale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#Applying GuassianBlur to remove noise
blur_img = cv2.GaussianBlur(src = gray_img, ksize = (5,5), sigmaY = 2, sigmaX=2)
# Applying Median Blur to remove noise
median_img = cv2.medianBlur(src = blur_img, ksize = 5)


# sharpened Image for better Edge Detection:
# sharpened  = original + (Original - Blur) * amount
sharpening_kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])
sharpened_img = cv2.filter2D(src = img, ddepth= -1, kernel = sharpening_kernel)
gray_sharpened_img = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2GRAY)

median_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2GRAY)
adaptive_blur_img =  cv2.adaptiveThreshold(src = median_img,
                                           maxValue= 255,   # max value to be given to the pixel after threshold
                                           adaptiveMethod= cv2.ADAPTIVE_THRESH_GAUSSIAN_C,     # adaptive method
                                           thresholdType= cv2.THRESH_BINARY_INV,  # binary threshold
                                           blockSize=9,   # neighborhood kernel size
                                           C=5         # constant value to be subtacted from mean or weighted mean
                                           )

cv2.rectangle(adaptive_blur_img, (0, 0), (270, 50), (0, 0, 0), -1)
cv2.putText(adaptive_blur_img, text='Adaptive Blur Image', org=(20, 30), color=(255, 255, 255),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, thickness=2, fontScale=0.8)



# Erosion
erosion_kernel = np.ones((3,3), np.uint8)
eroded_img = cv2.erode(src = adaptive_blur_img,
                       kernel = erosion_kernel,
                       iterations=1  # number of times erosion is applied
                       )
cv2.rectangle(eroded_img, (0, 0), (270, 50), (0, 0, 0), -1)
cv2.putText(eroded_img, text='Eroded Image', org=(20, 30), color=(255, 255, 255),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, thickness=2, fontScale=0.8)


# Dilation
dilation_kernel = np.ones((5,5), np.uint8)
dilated_img = cv2.dilate(src = adaptive_blur_img,
                         kernel = dilation_kernel,
                         iterations=1   # number of times erosion is applied
                         )
cv2.rectangle(dilated_img, (0, 0), (270, 50), (0, 0, 0), -1)
cv2.putText(dilated_img, text='Dilated Image', org=(20, 30), color=(255, 255, 255),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, thickness=2, fontScale=0.8)


#HoughCircles Method
circles = cv2.HoughCircles( image = adaptive_blur_img,
                           method = cv2.HOUGH_GRADIENT,
                           dp = 1,
                           minDist=1,   # min distance between centers of the circles
                           param1 = 100, # for edge Detection Higher Threshold
                           param2= 122,  # circles selecting threshold
                           minRadius= 5,  # min radius of circles
                           maxRadius= 600  # max radius of circles
                           )


output1 = img.copy()
if circles is not None:
    circles = np.round(circles[0,:]).astype("int")

# print(len(circles))
try:
   for (x,y,r) in circles:
       cv2.circle(output1, (x,y), r, (255,0,0), 1)
except:
   pass




try:
    # Biggest circle
    biggest_circle_index = np.where(circles[:,2] == np.max(circles[:,2], axis = 0)  )
    biggest_circle = circles[biggest_circle_index][0]
    print('Biggest Circle(x,y,r) obtained by HoughCircles method is: {}'.format(biggest_circle))
    # Smallest Circle
    smallest_circle_index = np.where(circles[:,2] == np.min(circles[:,2], axis = 0))
    smallest_circle =  circles[smallest_circle_index][0]
    print('Smallest Circle(x,y,r) obtained by HoughCircles method is:{}'.format(smallest_circle))

    # Centers Analysis
    x_coordinates = circles[:,0]
    y_coordinates = circles[:,1]

    # Mean
    x_mean = np.mean(x_coordinates)
    y_mean = np.mean(y_coordinates)


    # Standared Deviation
    Pixel_Wise_Standard_Deviation_x_HoughCircles_method = np.sqrt((np.sum(np.square(x_coordinates - x_mean)))/len(x_coordinates))
    Pixel_Wise_Standard_Deviation_y_HoughCircles_method = np.sqrt((np.sum(np.square(y_coordinates - y_mean)))/len(y_coordinates))

    print('Pixel Wise Standard Deviation x using HoughCircles method: {}'.format(Pixel_Wise_Standard_Deviation_x_HoughCircles_method ))
    print('Pixel Wise Standard Deviation y using HoughCircles method: {}'.format(Pixel_Wise_Standard_Deviation_y_HoughCircles_method ))


except:
    pass

cv2.rectangle(output1, (0, 0), (315, 50), (0, 0, 0), -1)
cv2.putText(output1, text='HoughCircles Method', org=(20, 30), color=(255, 255, 255),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, thickness=2, fontScale=0.8)




# Drawing Contours
cnts = cv2.findContours(adaptive_blur_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
output2 = img.copy()
cv2.drawContours(output2, cnts, -1, (0,255,0), 2)

cv2.rectangle(output2, (0, 0), (270, 50), (0, 0, 0), -1)
cv2.putText(output2, text='Contours', org=(20, 30), color=(255, 255, 255),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, thickness=2, fontScale=0.8)


# fitEllipse
output3 = img.copy()
ellipse_list = []
for cnt in cnts:
    if len(cnt)>5:
        ellipse = cv2.fitEllipse(cnt)
        ellipse_list.append(ellipse)


# cv2.fitElipse output: ((x_c, y_c), (2a, 2b), theta )
ellipse_selected = [ i  for i in ellipse_list   if   300<i[0][0]<380 and 370<i[0][1]<430  and (abs(i[1][0] - i[1][1])) < 10  ]
x_c_list = [ i[0][0]   for i in ellipse_selected]
y_c_list = [ i[0][1]   for i in ellipse_selected]


x_c_array = np.array(x_c_list)
y_c_array = np.array(y_c_list)

mean_x_c = sum(x_c_list)/len(x_c_list)
mean_y_c = sum(y_c_list)/len(y_c_list)

Pixel_Wise_Standard_Deviation_x_EllipseFit_method = np.sqrt(
    (np.sum(np.square(x_c_array - mean_x_c))) / len(x_c_array))
Pixel_Wise_Standard_Deviation_y_EllipseFit_method = np.sqrt(
    (np.sum(np.square(y_c_array - mean_y_c))) / len(y_c_array))

print('________________________________________________________________________________________________')
print('Pixel Wise Standard Deviation x using FitEllipse method: {}'.format(Pixel_Wise_Standard_Deviation_x_EllipseFit_method))
print('Pixel Wise Standard Deviation y using FitEllipse method: {}'.format(Pixel_Wise_Standard_Deviation_y_EllipseFit_method))


for i in ellipse_selected:
    cv2.ellipse(output3, i, (255, 0, 0), 2)

cv2.rectangle(output3, (0, 0), (270, 50), (0, 0, 0), -1)
cv2.putText(output3, text='fitEllipse Method', org=(20, 30), color=(255, 255, 255),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, thickness=2, fontScale=0.8)


# # minEnclosingCircle
# output4 = img.copy()
# for cnt in cnts:
#   if len(cnt) >= 5:
#     (x,y),radius = cv2.minEnclosingCircle(cnt)
#     center = (int(x),int(y))
#     radius = int(radius)
#     cv2.circle(output4,center,radius,(0,255,0),2)


# Showing Stacked Images
stacked_images = stackImages(0.4, ([adaptive_blur_img, eroded_img, dilated_img],
                                   [output1, output2, output3]
                                   ))
cv2.imshow('Stacked Images', stacked_images)


cv2.waitKey(0)


