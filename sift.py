import cv2
import numpy as np
import os

"""
SIFT features detection code - when passed an image to this module, it
detects and outputs the sift keypoints. The detect function is the driver
function to detect the keypoints and return a list of
opencv keypoints datatypes
"""


class SIFT():
    def __init__(self, image):
        self.num_octaves = 4
        self.num_intervals = 5
        self.sigma_preblur = 1.0
        self.sigma_increment = 1.4142
        self.sigma_factor = 0.5
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # A single row in this represents every single level of octave
        self.octaves = [[None] * self.num_intervals for _ in
                        range(self.num_octaves)]
        # A single row in this represents a single sigma
        # list at every level of octave
        self.sigma = [[None] * self.num_intervals for _ in
                      range(self.num_octaves)]
        # for DOG octave
        self.DOG_octaves = [[None] * (self.num_intervals - 1) for _ in
                            range(self.num_octaves)]
        image = cv2.GaussianBlur(image, (0, 0), self.sigma_factor)
        # blur the image
        double_image = cv2.GaussianBlur(image, (0, 0), self.sigma_preblur)
        self.sigma[0][0] = 0.70717
        self.current_image = double_image.copy()
        self.global_keypoints = []
        return

    """
    Step 1 Scale space, octaves and blurs generations
    for gaussian
    """

    # define a scale space for gaussian
    def scale_space(self,):
        self.get_octaves()
        return

    # Find blurs in all the octaves
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
            image = cv2.resize(self.octaves[i][self.num_intervals-1],
                               (current_width, current_height))
            self.current_image = image.copy()
        return

    # compute blurs for a single octave
    def get_single_octave(self, sigma_current, row):
        image = self.current_image
        for i in range(self.num_intervals):
            # apply gaussian
            image = cv2.GaussianBlur(image, (0, 0), sigma_current)
            # set image to octaves[row,i]
            self.octaves[row][i] = image
            self.sigma[row][i] = sigma_current
            sigma_current = sigma_current * self.sigma_increment
            if i > 0:
                self.DOG_octaves[row][i-1] = self.octaves[row][i] -
                self.octaves[row][i-1]
        return

    # Driver function to call step functions
    def detect(self):
        self.scale_space()
        self.detect_keypoints()
        print("detected")
        return self.global_keypoints

    # helper function to show octaves
    def show_octaves(self, octaves):
        for i in octaves:
            for j in i:
                self.show_image(j)
        return

    # helper function show image
    def show_image(self, image):
        cv2.imshow("image", image)
        cv2.waitKey(0)
        return

    """
    Step 2 Keypoints localization to locate keypoints in
    the different blurred images in all the octaves
    """

    # detect keypoints driver function
    def detect_keypoints(self,):
        self.get_maxima_minima()
        return

    # get extrema/ keypoints for all octaves
    def get_maxima_minima(self):
        # a row represents a single octave and col represents
        # the 2 DOG images in this case
        self.keypoints_octaves = []
        # get from single octave
        for i in range(self.num_octaves):
            self.keypoints_octaves.
            append(self.get_maxima_minima_single_octave(i))
        return

    # compute keypoints for one octave
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
                    neighbours = self.get_neighbours(row, col, prev_scale,
                                                     current_scale, next_scale)
                    # calculate hessian matrix
                    hessian_matrix = self.get_hessian_matrix(row, col,
                                                             prev_scale,
                                                             current_scale,
                                                             next_scale)
                    # check local maxima and minima
                    is_extrema = self.check_extrema(current_scale[row][col],
                                                    neighbours)
                    # if pixel is local extrema and it's hessian
                    # is invertible then proceed
                    if is_extrema is True and
                    np.linalg.det(hessian_matrix) != 0:
                        # generate a keypoint object
                        subpixel_value, delta_X = self.
                        get_subpixel_maxima_minima(row, col, prev_scale,
                                                   current_scale,
                                                   next_scale,
                                                   hessian_matrix)
                        check_for_edge = self.get_edge_check(hessian_matrix)
                        if subpixel_value > 0.03 and check_for_edge is False:
                            single_keypoint = cv2.KeyPoint()
                            single_keypoint.pt = (int(col * np.power(2, i)),
                                                  int(row * np.power(2, i)))
                            single_keypoint.octave = np.power(2, i)
                            single_keypoint.response = int(subpixel_value)
                            single_keypoint.angle = 196.0
                            # append to the global keypoints list
                            # to be given as output
                            self.global_keypoints.append(single_keypoint)
                            image[row, col] = 255
            keypoints_frame.append(image)
        return keypoints_frame

    # return the list of 26 neighbours of a pixel in current,
    # previous and next scale
    def get_neighbours(self, row, col, prev_scale, current_scale,
                       next_scale):
        neighbours = []
        current_row = row - 1
        current_col = col - 1
        # check  top left most neighbour
        neighbours.append(prev_scale[current_row][current_col])
        neighbours.append(current_scale[current_row][current_col])
        neighbours.append(next_scale[current_row][current_col])
        # check top middle most neighbour
        neighbours.append(prev_scale[current_row][current_col+1])
        neighbours.append(current_scale[current_row][current_col+1])
        neighbours.append(next_scale[current_row][current_col+1])
        # check top right neighbour
        neighbours.append(prev_scale[current_row][current_col+2])
        neighbours.append(current_scale[current_row][current_col+2])
        neighbours.append(next_scale[current_row][current_col+2])
        # check right most neighbour
        neighbours.append(prev_scale[current_row+1][current_col+2])
        neighbours.append(current_scale[current_row+1][current_col+2])
        neighbours.append(next_scale[current_row+1][current_col+2])
        # check bottom right most neighbour
        neighbours.append(prev_scale[current_row+2][current_col+2])
        neighbours.append(current_scale[current_row+2][current_col+2])
        neighbours.append(next_scale[current_row+2][current_col+2])
        # check bottom most neighbour
        neighbours.append(prev_scale[current_row+2][current_col+1])
        neighbours.append(current_scale[current_row+2][current_col+1])
        neighbours.append(next_scale[current_row+2][current_col+1])
        # check bottom left most neighbour
        neighbours.append(prev_scale[current_row+2][current_col])
        neighbours.append(current_scale[current_row+2][current_col])
        neighbours.append(next_scale[current_row+2][current_col])
        # check left most neighbour
        neighbours.append(prev_scale[current_row+1][current_col])
        neighbours.append(current_scale[current_row+1][current_col])
        neighbours.append(next_scale[current_row+1][current_col])
        # append the middle pixel in prev and next scale
        neighbours.append(prev_scale[row][col])
        neighbours.append(next_scale[row][col])
        return neighbours

    # helper function for get_neighbours function
    # to calculate direction to find neighbours
    def get_direction(self, current_row, current_col, row, col, direction):
        if direction == "right" and current_col == col + 1:
            direction = "down"
        elif direction == "down" and current_row == row + 1:
            direction = "left"
        elif direction == "left" and current_col == col - 1:
            direction = "up"
        return direction

    # check if pixel is maximum or minimum of all of its neighbours
    def check_extrema(self, num, neighbours):
        # check if minima
        minima = all(i > num for i in neighbours)
        if minima is False:
            # check if maxima
            maxima = all(i < num for i in neighbours)
            return maxima
        return minima

    # get subpixel maxima and minima value of the pixel.
    def get_subpixel_maxima_minima(self, row, col, prev_scale,
                                   current_scale,
                                   next_scale,
                                   hessian_matrix):
        # multiply negative of inverse of hessian matrix with gradient matrix
        # formulae from reference
        # https://www.ipol.im/pub/art/2014/82/article_lr.pdf
        gradient_matrix = self.get_gradient_matrix(row, col, prev_scale,
                                                   current_scale, next_scale)
        inverse_hessian = -np.linalg.inv(hessian_matrix)
        delta_X = np.matmul(inverse_hessian, gradient_matrix)
        old_val = current_scale[row, col]
        sub_pixel_value = current_scale[row, col] + ((1 / 2) *
                                                     np.matmul(
                                                               gradient_matrix.
                                                               transpose(),
                                                               delta_X))
        # normalize values between 0 and 1
        sub_pixel_value = sub_pixel_value / 255
        return np.abs(sub_pixel_value), delta_X

    # calculate hessian matrix
    def get_hessian_matrix(self, row, col, prev_scale,
                           current_scale, next_scale):
        # formulae from reference
        # https://www.ipol.im/pub/art/2014/82/article_lr.pdf
        dxx = (current_scale[row, col+1] + current_scale[row, col-1] -
               2 * current_scale[row, col])
        dyy = (current_scale[row+1, col] + current_scale[row-1, col] -
               2 * current_scale[row, col])
        dscale_scale = (next_scale[row, col] + prev_scale[row, col] -
                        2 * current_scale[row, col])
        dxy = (current_scale[row+1, col+1] - current_scale[row-1, col+1] -
               current_scale[row+1, col-1] +
               current_scale[row-1, col-1]) / 4
        dx_scale = (next_scale[row, col+1] - prev_scale[row, col+1] -
                    next_scale[row, col-1] +
                    prev_scale[row, col-1]) / 4
        dy_scale = (next_scale[row+1, col] - prev_scale[row+1, col] -
                    next_scale[row-1, col] +
                    prev_scale[row-1, col]) / 4
        hessian_matrix = np.array([[dxx, dxy, dx_scale], [dxy, dyy, dy_scale],
                                  [dx_scale, dy_scale, dscale_scale]])
        return hessian_matrix

    # calculate gradient matrix
    def get_gradient_matrix(self, row, col, prev_scale,
                            current_scale, next_scale):
        # formulae from reference
        # https://www.ipol.im/pub/art/2014/82/article_lr.pdf
        dx = (current_scale[row, col+1] - current_scale[row, col-1]) / 2
        dy = (current_scale[row+1, col] - current_scale[row-1, col]) / 2
        dscale = (next_scale[row, col] - prev_scale[row, col]) / 2
        gradient_matrix = np.array([[dx, dy, dscale]])
        gradient_matrix = gradient_matrix.transpose()
        return gradient_matrix

    # check if pixel is an edge
    def get_edge_check(self, hessian_matrix):
        dxx = hessian_matrix[0, 0]
        dxy = hessian_matrix[0, 1]
        dyy = hessian_matrix[1, 1]
        # hessian_matrix = hessian_matrix / 255
        sub_hessian_matrix = np.array([[dxx, dxy], [dxy, dyy]])
        determinant_hessian = np.linalg.det(sub_hessian_matrix)
        trace_hessian = np.trace(sub_hessian_matrix)
        # if determinant less than or equal to 0 break loop
        if determinant_hessian <= 0:
            return True
        ratio = (np.square(trace_hessian)) / determinant_hessian
        r = 10
        # if an edge then return true
        if ratio > ((np.square(r + 1)) / r):
            return True
        # if not an edge return False
        return False
