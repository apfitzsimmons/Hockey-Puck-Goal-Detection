import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def histogram(img):
    #converts color image matrix to greyscale and displays histogram of intensity values
    cv.imshow('greyscale',img)
    cv.waitKey(0)
    #3B
    #calculate and display image intensity histogram
    plt.hist(img.ravel(),256,[0,256])
    plt.show()


def removeBoarder(img):
    '''removes black boarder from image so it is not detected as contour'''

    column_dim = int(img.shape[1])
    row_dim = int(img.shape[0])

    for y in range(0,row_dim):
        for x in range(0,column_dim):
            if y<5 or y>(row_dim-5) or x<5 or x>(column_dim-5):
                img[y,x]=0
    return img

def determineGoal(puck_x,puck_y,row_dim,goal_x,polyFeatures,goalLineModel):
    #set features of polynomial regression to degree = 2 (2nd degree polynomial x^2)
    #generate x^2 feature values for regression on puck x values
    puck_poly = PolynomialFeatures(degree = 2, include_bias=False)
    puck_poly_features = puck_poly.fit_transform(puck_x.reshape(-1,1))

    #predict goal line location for plot
    goal_y_predicted = goalLineModel.predict(polyFeatures)

    #predict y values based on feature data
    puck_y_predicted = goalLineModel.predict(puck_poly_features)

    plt.figure(figsize=(20, 6))
    plt.title("Polynomial Regression – Goal Line Boundary and Puck", size=16)
    plt.ylim(row_dim,0)
    plt.scatter(puck_x, puck_y, c='black')
    plt.plot(goal_x, goal_y_predicted, c="red")
    plt.show()
    
    #compare predicted values to actual puck coordinate values, if actual is less than predicted (higher on the image above the goal line is a smaller y val)
    #then exit loop and return False (no goal)
    for i in range(0,len(puck_y_predicted)):
        y_predicted = puck_y_predicted[i]
        y_actual = puck_y[i]
        if y_actual <= y_predicted:
            return False
    return True

def getPuckCoordinates(img):

    #find row and column dimensions
    column_dim = int(img.shape[1])
    row_dim = int(img.shape[0])

    #create empty dictionary to hold pixel coordinates that were previously segmented
    edge_dictionary = {}    
    #iterate across R matrix to update dictionary key and values with key as x and value as y
    #doing so in ascending row order results in a coordinate representation of the bottom of the goal line
    #starting at 30 removes any potential red values above the goal line
    for y in range(0,row_dim):
        for x in range(0,column_dim):
            B = img[y,x][0]
            G = img[y,x][1]
            R = img[y,x][2]

            if B == 0 and G == 255 and R == 0 and x not in edge_dictionary:
                edge_dictionary[x]=y
    
    #pull dictionary keys and values out into separate x and y lists
    xlist = list(edge_dictionary.keys())
    ylist = list(edge_dictionary.values())
    #sort xlist by ascending value and return index location
    xidx = np.argsort(xlist)
    
    #create empty lists to populate with ordered x and y coordinates
    xordered = []
    yordered = []
    #iterate across x and y lists using xidx values to generate ordered lists of coordinates from smallest to largest x value
    for idx in xidx:
        xordered.append(xlist[idx])
        yordered.append(ylist[idx])
    
    #convert x and y ordered lists to arrays
    xvector = np.array(xordered)
    yvector = np.array(yordered)

    return xvector,yvector

def findPuck(img):
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #load image into HSV space and blur
    #HSV_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #HSV_img = cv.blur(HSV_img,(3,3))
    #find row and column dimensions
    column_dim = int(grey_img.shape[1])
    row_dim = int(grey_img.shape[0])
    
    #create black and white image based on threshold value
    grey_neg_img = abs(255-grey_img)
    grey_neg_img = removeBoarder(grey_neg_img)
    ret, thresh = cv.threshold(grey_neg_img, 220, 255, cv.THRESH_BINARY)
    cv.namedWindow('B/W_Thresholded_Image')        # Create a named window
    cv.moveWindow('B/W_Thresholded_Image', 1500,50)  # Move it to (1500,50)
    cv.imshow('B/W_Thresholded_Image',thresh)
    cv.waitKey(0)
    
    #create BGR representation of black and white image
    b_w_BGR = cv.cvtColor(thresh.copy(), cv.COLOR_GRAY2BGR)
    
    #find contours in b/w image
    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    img_copy = img.copy()
    #select max contour as c (hopefully the puck)
    c = max(contours, key = cv.contourArea)
    #overlay the max contour outline onto the b/w BGR image as green outline
    contour_img = cv.drawContours(image=b_w_BGR, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    
    #display contour outline on b/w image
    cv.namedWindow('Puck_Contour_Outline')        # Create a named window
    cv.moveWindow('Puck_Contour_Outline', 1500,210)  # Move it to (1500,210)
    cv.imshow('Puck_Contour_Outline', contour_img)
    cv.waitKey(0)
    cv.destroyWindow('B/W_Thresholded_Image')
    cv.destroyWindow('Puck_Contour_Outline')

    return contour_img

def findGoalLine(img):
    '''uses color segmentation on Hue component of HSV image to segment red goal line from other area
    based on 'found' red pixel values, model boundary line for goal using polynomial regression'''
    
    #load image into HSV space and blur
    HSV_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    HSV_img = cv.blur(HSV_img,(5,5))
    #find row and column dimensions
    column_dim = int(HSV_img.shape[1])
    row_dim = int(HSV_img.shape[0])
    #create base structure for h component values
    h_img = np.zeros((row_dim,column_dim),dtype=np.uint8)

    #plot h component values into h_img from HSV set
    for y in range(0,row_dim):
        for x in range(0,column_dim):
            hsv_val = HSV_img[y][x]
            h = hsv_val[0]
            h_img[y,x] = h

    #hue segmentation limits
    red1 = [0,20]
    red2 = [160,179]
    #apply red segmentation limits to image h component
    R1 = (255 * ((h_img >= red1[0])&(h_img <= red1[1]))).astype(np.uint8)
    R2 = (255 * ((h_img >= red2[0])&(h_img <= red2[1]))).astype(np.uint8)
    R = R1+R2
    cv.namedWindow('Hue_Segmentation_Red_Objects_Only')        # Create a named window
    cv.moveWindow('Hue_Segmentation_Red_Objects_Only', 1500,210)  # Move it to (1500,210)
    cv.imshow('Hue_Segmentation_Red_Objects_Only',R)
    cv.waitKey(0)
    cv.destroyWindow('Hue_Segmentation_Red_Objects_Only')
    cv.destroyWindow('Cropped_Image')

    #create empty dictionary to hold pixel coordinates that were previously segmented
    edge_dictionary = {}    
    #iterate across R matrix to update dictionary key and values with key as x and value as y
    #doing so in ascending row order results in a coordinate representation of the bottom of the goal line
    #starting at 30 removes any potential red values above the goal line
    for y in range(30,row_dim):
        for x in range(0,column_dim):
            if R[y,x] == 255:
                edge_dictionary[x]=y
    #pull dictionary keys and values out into separate x and y lists
    xlist = list(edge_dictionary.keys())
    ylist = list(edge_dictionary.values())
    #sort xlist by ascending value and return index location
    xidx = np.argsort(xlist)
    
    #create empty lists to populate with ordered x and y coordinates
    xordered = []
    yordered = []
    #iterate across x and y lists using xidx values to generate ordered lists of coordinates from smallest to largest x value
    for idx in xidx:
        xordered.append(xlist[idx])
        yordered.append(ylist[idx])
    
    #convert x and y ordered lists to arrays
    xvector = np.array(xordered)
    yvector = np.array(yordered)

    #set features of polynomial regression to degree = 2 (2nd degree polynomial x^2)
    #generate x^2 feature values for regression
    poly = PolynomialFeatures(degree = 2, include_bias=False)
    poly_features = poly.fit_transform(xvector.reshape(-1,1))
    #call linear regression function and fit to data
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, yvector)
    #predict y values based on feature data
    y_predicted = poly_reg_model.predict(poly_features)
    
    #plot polynomial regression line over original goal line boundary coordinates
    plt.figure(figsize=(20, 6))
    plt.title("Polynomial Regression – Goal Line Boundary", size=16)
    plt.ylim(row_dim,0)
    plt.scatter(xvector, yvector)
    plt.plot(xvector, y_predicted, c="red")
    plt.show()

    return row_dim,xvector,poly_features,poly_reg_model

def cropImage(img):
    #crops image to remove all non-goal area information
    cropped_image = img[375:499]
    cv.namedWindow('Cropped_Image')        # Create a named window
    cv.moveWindow('Cropped_Image', 1500,50)  # Move it to (1500,50)
    cv.imshow('Cropped_Image',cropped_image)
    cv.waitKey(0)
    return cropped_image

def resizeImage(img):
    #takes color image matrix and resizes image to 500 by X pixels (maintains aspect ratio)
    #returns scaled image
    scaleFactor = img.shape[0]/500
    width = int(img.shape[1] / scaleFactor)
    height = int(500)
    dim = (width, height)
    r_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    #cv.imshow('r_img',r_img)
    #cv.waitKey(0)
    return r_img

def loadImage(filename):
    #takes file name stored in same folder as program and returns color image matrix representation
    img = cv.cvtColor(cv.imread(filename),cv.IMREAD_COLOR)
    cv.namedWindow('Original')        # Create a named window
    cv.moveWindow('Original', 1500,50)  # Move it to (1500,50)
    cv.imshow('Original',img)                                               #displays original image
    cv.waitKey(0)
    cv.destroyWindow('Original')
    return img

def hockeyGoalDetection(filename):
    img =  loadImage(filename)                                              #loads file into image matrix
    r_img = resizeImage(img)                                                #resize image for consistant processing
    cropped_img = cropImage(r_img)                                          #crop image to only the goal area
    row_dim,goal_x,polyFeatures,goalLineModel = findGoalLine(cropped_img)   #detect goal line and use regression to interpret goal line boundary
    puck_img = findPuck(cropped_img)                                        #detect puck in image and interpret location
    puck_x,puck_y = getPuckCoordinates(puck_img)                            #interpret coordinates of green puck outline in image
    if determineGoal(puck_x,puck_y,row_dim,goal_x,polyFeatures,goalLineModel):  #based on goal line model and actual puck coordinates, determine if all of puck is within goal line
        print('GOAL')
    else:
        print('NO GOAL')
    
    '''
    for this goal detection program to be implemented for practical use
    
    * camera angle for these photos is meant more to capture an artistic and dynamic angle of goal scoring
        for image processing, a strip array of cameras along the top of the crossbar looking down would provide the best angle for processing goals since everything could be interpreted as a 2D image
        camera array would remove distortion caused by the camera lens
    * puck object would likely have to be a 'trained' set of points that you would compare contour objects to and select the most similar contour object to the training data
        currently vulnerable to the max contour being something other than the puck if there are other large, dark objects in the image
    * does not handle obstruction well
        majority of puck needs to be visible for this program to work. could implement higher level puck tracking vision system that is interpreting puck position
        all over the ice. could tie in with goal detection for multiple levels of verification.
    '''
    

hockeyGoalDetection('goal.jpg')