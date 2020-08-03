import cv2
import numpy as np
import os

class SIFT():
    def __init__(self, image):
        self.num_octaves = 4
        self.num_intervals = 5
        self.sigma_preblur = 1.0
        self.sigma_increment = 1.4142
        self.sigma_factor=0.5
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # A single row in this represents every single level of octave
        self.octaves = [[None] * self.num_intervals for _ in range(self.num_octaves)]
        # A single row in this represents a single sigma list at every level of octave
        self.sigma = [[None] * self.num_intervals for _ in range(self.num_octaves)]
        # for DOG octave
        self.DOG_octaves = [[None] * (self.num_intervals - 1) for _ in range(self.num_octaves)]
        image = cv2.GaussianBlur(image, (0, 0), self.sigma_factor)
        # Double the dimensions of the image
        double_image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), 
                                  interpolation = cv2.INTER_AREA)
        # blur the image
        double_image = cv2.GaussianBlur(double_image, (0, 0), self.sigma_preblur)
        #sigma_current = sigma_increment * sigma_factor
        self.sigma[0][0] = 0.70717
        self.current_image = double_image.copy()
        #self.show_image(self.current_image)
        return
    
    
    def scale_space(self,):
        self.get_octaves()
        return
    
    
    def get_octaves(self,):
        # Generate octaves 4 octaves and 5 blurs in one octave
        for i in range(self.num_octaves):
            if i != 0:    
                sigma_current = self.sigma[i-1][0] * self.sigma_increment
            else:
                sigma_current = 0.707107
            # get single octave 
            self.get_single_octave(sigma_current, i)
            current_width = int(self.current_image.shape[1] / 2)
            current_height = int(self.current_image.shape[0] / 2)
            image = cv2.resize(self.octaves[i][self.num_intervals-1], (current_width, current_height))  
            self.current_image = image.copy()
        #self.show_octaves(self.DOG_octaves)   
        return 


    def get_single_octave(self, sigma_current, row):
        image = self.current_image
        for i in range(self.num_intervals):
            # apply gaussian
            image = cv2.GaussianBlur(image, (0, 0), sigma_current)
            # set image to octaves[row,i]
            self.octaves[row][i] = image
            self.sigma[row][i] = sigma_current
            sigma_current = sigma_current * self.sigma_increment
            if i>0:
                self.DOG_octaves[row][i-1] = self.octaves[row][i] - self.octaves[row][i-1]
        return 
    
    
    def detect(self):
        self.scale_space()
        self.detect_keypoints()
        return
    
    
    def show_octaves(self, octaves):
        for i in octaves:
            for j in i:
                self.show_image(j)
        return


    def show_image(self, image):
        cv2.imshow("image", image)
        cv2.waitKey(0)
        return
    
    """ 
    Step 2 Keypoints localization
    """
    
    def detect_keypoints(self,):
        self.get_maxima_minima()
        return
    
    
    def get_maxima_minima(self):
        # a row represents a single octave and col represents the 2 DOG images in this case 
        self.keypoints_octaves = []
        # get from single octave 
        for i in range(self.num_octaves):  
            self.keypoints_octaves.append(self.get_maxima_minima_single_octave(i))
        self.show_octaves(self.keypoints_octaves)   
        return        
    
    
    def get_maxima_minima_single_octave(self, i):
     
        # skip first and last scale
        keypoints_frame = []
        for scale in range(1, self.num_intervals - 2):
            prev_scale = self.DOG_octaves[i][scale - 1]
            current_scale = self.DOG_octaves[i][scale] 
            next_scale = self.DOG_octaves[i][scale + 1]
            # get single pixel
            row_bound = current_scale.shape[0]
            col_bound = current_scale.shape[1]
            image = np.zeros((current_scale.shape[0], current_scale.shape[1]))
            # skip first and last row and first col and last col
            for row in range(1, row_bound-1):
                for col in range(1, col_bound-1):
                    # get neighbours of current pixel.
                    neighbours = self.get_neighbours(row, col, prev_scale, current_scale, next_scale)
                    # check maxima and minima
                    is_extrema = self.check_extrema(current_scale[row][col], neighbours)
                    if is_extrema == True:
                        image[row, col] = 255
                #print("octave:" + str(i) + "scale" + str(scale) + " row" + str(row))
            #self.show_image(image)
            keypoints_frame.append(image)
        return keypoints_frame
    
    def get_neighbours(self, row, col, prev_scale, current_scale, 
                       next_scale):
        neighbours = []
        row_start = row - 1
        col_start = col - 1
        current_row = row_start
        current_col = col_start
        direction = "right"
        flag = True
        while(flag):
            neighbours.append(prev_scale[current_row][current_col])
            neighbours.append(current_scale[current_row][current_col])
            neighbours.append(next_scale[current_row][current_col])
            direction = self.get_direction(current_row, current_col, row, col, direction)
            if direction == "right":
                current_col = current_col + 1
            if direction == "down":
                current_row = current_row + 1
            if direction == "left":
                current_col = current_col - 1
            if direction == "up":
                current_row = current_row - 1
            # check flag
            if current_row == row_start and current_col == col_start:
                flag = False
        neighbours.append(prev_scale[row][col])
        neighbours.append(next_scale[row][col])
        return neighbours
    
    
    def get_direction(self,current_row, current_col, row, col, direction):
        if direction == "right" and current_col == col + 1:
            direction = "down"
        elif direction == "down" and current_row == row + 1:
            direction = "left"
        elif direction == "left" and current_col == col - 1:
            direction = "up"
        return direction
        
    
    def check_extrema(self, num, neighbours):
        minima = all(i > num for i in neighbours)
        if minima == False:    
            maxima = all(i < num for i in neighbours)
            return maxima
        return minima
    

    def get_subpixel_maxima_minima(self,):
        
        return


image = cv2.imread("images/car1.jpg")
sift = SIFT(image)
sift.detect()
  
    
    

