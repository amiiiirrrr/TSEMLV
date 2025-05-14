
import numpy as np
import cv2
import math

class measureTumor_V1:
    def __init__(self, args):
        self.args = args

    def minAreaRect_SI(self, cnt, mask_biggestSI, height, width):
        '''
        https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        (center(x, y), (width, height), angle of rotation) = rect

        references:
            https://stackoverflow.com/questions/28710337/opencv-lines-passing-through-centroid-of-contour-at-given-angles
        '''

        self.height = height
        self.width = width
        self.cnt = cnt

        mask_SI = cv2.cvtColor(mask_biggestSI, cv2.COLOR_BGR2GRAY)
        img = mask_biggestSI.copy()
        x,y,w,h = cv2.boundingRect(self.cnt)
        
        # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        # compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(self.cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # draw minimum area rectangle (rotated rectangle)
        # img = cv2.drawContours(img,[box],0,(0,255,255),2)

        # rank based on their y. after sorting extract y1 and y0. the line between these two dots is what we are looking for
            
        ref = mask_SI.copy()
        # cv2.drawContours(ref, contours, 0, 255, 1)
        cv2.drawContours(ref, self.cnt, 0, 255, 1)
        # cv2.imwrite(os.path.join(self.path_save, 'ref.png'), ref)

        tmp = np.zeros_like(mask_SI)

        ((c_x, c_y), (width_shape, height_shape), angle) = rect
        
        # # print("angleangleangleangle", angle)
        # if angle < 45:
        #     theta = (90-angle) * np.pi/180.0
        # if angle >= 45:
        #     theta = (180-angle) * np.pi/180.0
            
        # sorted_box = sorted(box, key=lambda box: box[1])
        # tmp = self.draw_parallel_line(tmp, p1=[c_x, c_y], p2=sorted_box[1], p3=sorted_box[0])

        tmp = self.find_vertical_dots(tmp, p=[c_x, c_y], rectangle=box)
        # cv2.imwrite(os.path.join(self.path_save, 'tmp.png'), tmp)
        # cv2.line(tmp, (int(c_x), int(c_y)),
        #    (int(int(c_x)+np.cos(theta)*self.width),
        #     int(int(c_y)-np.sin(theta)*self.height)), 255, 1)
        
        (row, col) = np.nonzero(np.logical_and(tmp, ref))

        tmp_and = np.logical_and(tmp, ref)
        tmp_rgb = cv2.cvtColor(np.uint8(tmp_and*255), cv2.COLOR_GRAY2BGR)
        # cv2.circle(tmp_rgb,(int(c_x), int(c_y)), radius=0, color=(0, 0, 255), thickness=3)
        # cv2.circle(tmp_rgb,(col[0],row[0]), radius=0, color=(0, 0, 255), thickness=3)
        # cv2.circle(tmp_rgb,(col[-1],row[-1]), radius=0, color=(0, 0, 255), thickness=3)
        # cv2.imwrite(os.path.join(self.path_save, 'tmp_rgb.png'), tmp_rgb)
        
        out_visualize = img + tmp_rgb
        cv2.line(out_visualize, (col[-1],row[-1]), (col[0],row[0]), (0, 0, 255), 10)
        # cv2.imwrite(os.path.join(self.path_save, 'out_visualize.png'), out_visualize)
        ############################################ calculate the length of the SI ############################################
        # (col[0],row[0])
        # (c_x, c_y)
        # length1 = 2 * (np.sqrt((c_x - col[0]) ** 2 + (c_y - row[0]) ** 2))
        length = (np.sqrt((col[-1] - col[0]) ** 2 + (row[-1] - row[0]) ** 2))
        # print("length1", length1)
        # print("length2", length2)

        return length, rect, out_visualize
    
    def minAreaRect_SI2(self, mask_biggestSI):
        '''
        https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        (center(x, y), (width, height), angle of rotation) = rect

        references:
            https://stackoverflow.com/questions/28710337/opencv-lines-passing-through-centroid-of-contour-at-given-angles
        '''
        mask_SI = cv2.cvtColor(mask_biggestSI, cv2.COLOR_BGR2GRAY)
        # compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(self.cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        
        ref = mask_SI.copy()
        # cv2.drawContours(ref, contours, 0, 255, 1)
        cv2.drawContours(ref, self.cnt, 0, 255, 1)

        tmp = np.zeros_like(mask_SI)

        ((c_x, c_y), (width_shape, height_shape), angle) = rect

        tmp = self.find_vertical_dots(tmp, p=[c_x, c_y], rectangle=box)
        
        (row, col) = np.nonzero(np.logical_and(tmp, ref))
        
        point1 = (col[0],row[0])
        point2 = (col[-1],row[-1])

        return point1, point2
    
    def find_vertical_dots(self, image, p, rectangle):
        '''
        get a point and a rectangle
        rectangle contains 4 points that we need to find two vertical lines.
        first we find the closest point (out of rectangle points) to the image center
        then we create three lines from the closest point to the other points.
        then we try to find two vertical lines. then we have these two vertical lines
        and then we can find the smallest line. then we have p1 and p2
        '''
        # print("rectanglerectanglerectanglerectangle", rectangle)
        # Calculate the center of the image
        center_x, center_y = self.width / 2, self.height / 2

        # Initialize variables to track the closest point and its distance
        closest_point = None
        closest_distance = float('inf')  # Initialize with positive infinity

        # Iterate through the four dots and calculate distances
        for dot in rectangle:
            x, y = dot
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Check if this point is closer than the previous closest point
            if distance < closest_distance:
                closest_point = dot
                closest_distance = distance
        
        # print("closest_pointclosest_pointclosest_pointclosest_point", closest_point)
        # Create a new list without the specified element
        new_rectangle = [point for point in rectangle if not np.array_equal(point, closest_point)]
        # print("new_rectanglenew_rectanglenew_rectanglenew_rectangle", new_rectangle)

        
        # Initialize variables to store the perpendicular lines
        perpendicular_lines = []

        closest_point = (x, y) = tuple(closest_point)

        dot1, dot2 = self.find_perpendicular_dots(x, y, new_rectangle)
        
        p2, p3 = self.find_smallest_line(image, closest_point, dot1, dot2)

        # print("perpendicular_lines", perpendicular_lines)
        image = self.draw_parallel_line(image, p, p2, p3)

        return image
    
    def find_smallest_line(self, image, closest_point, dot1, dot2):
        '''
        take three point. we create two lines between central point to dot1 and dot2
        then we try to find the smalest line.
        '''
        len_side1 = np.sqrt((closest_point[0] - dot1[0])**2 + (closest_point[1] - dot1[1])**2)  
        len_side2 = np.sqrt((closest_point[0] - dot2[0])**2 + (closest_point[1] - dot2[1])**2) 

        # we choose the smaller line because it is parallel to the width
        if len_side1 < len_side2:
            p2 = closest_point
            p3 = dot1 
        else:
            p2 = closest_point
            p3 = dot2 
        
        return p2, p3
    
    def find_perpendicular_dots(self, x, y, dots):
        '''
        To find the two dots that create the approximately perpendicular lines with the given dot (x, y), you can use the concept of slope.
        Here's a step-by-step approach to solve this problem:
        Calculate the slope between the given dot (x, y) and each of the other three dots using the formula: slope = (y2 - y1) / (x2 - x1).
        Identify the two slopes that are approximately perpendicular. Perpendicular lines have slopes that are negative reciprocals of each other.
        So, look for two slopes that are close to each other and have a product close to -1.
        Once you have identified the two slopes, the corresponding dots will be the ones that have those slopes.
        '''
        min_diff = math.inf
        dot1 = None
        dot2 = None
        # print("dotsdotsdotsdotsdots", dots)
        slopes = []
        for dot in dots:
            slope = (dot[1] - y) / (dot[0] - x + 0.00001)
            slopes.append(slope)

        # print("closest", (x, y))
        # print("dots", dots)
        # print("slopes", slopes)
        # # if the rectangle is horizontal! 
        # if -math.inf in slopes:
        #     dot1 = dots[slopes.index(-math.inf)]
        #     dot2 = dots[slopes.index(0.0)]

        # else:
            # Find the two slopes that are approximately perpendicular
            # print("Hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        min_angle_diff = float('inf')
        for i in range(len(slopes)):
            for j in range(i + 1, len(slopes)):
                angle_diff = np.abs(np.arctan2(slopes[j], 1) - np.arctan2(slopes[i], 1))
                if angle_diff < np.pi / 2:
                    angle_diff = np.pi - angle_diff  # Correct the angle difference for obtuse angles
                angle_diff = np.abs(angle_diff)  # Ensure the angle difference is non-negative

                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    dot1 = dots[i]
                    dot2 = dots[j]

        # print("dot1", dot1)
        # print("dot2", dot2)
        return dot1, dot2
    
    def draw_parallel_line(self, image, p1, p2, p3):
        '''
        get 3 points. 
        p1: center point
        p2, p3: start and end point of another line
        output: draw a line which cross the p1 and is parallel to the line of p2 and p3
        '''
        # Define the points
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        # Calculate the direction vector of the line connecting p1 and p2
        direction_vector = p3 - p2

        # # Calculate the perpendicular vector to the direction vector
        # perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Normalize the perpendicular vector
        # normalized_perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)

        normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Calculate the start and end points of the parallel line
        start_point = p1 - normalized_direction_vector * self.width  # Adjust the length of the line as needed
        end_point = p1 + normalized_direction_vector * self.height

        # Draw the parallel line on an image
        cv2.line(image, tuple(start_point.astype(int)), tuple(end_point.astype(int)), 255, 1)  # Draw the line
        return image
    
    def find_distances_diameters(self, pixels_SI, avg_depth_tumor, avg_depth_SI, diameter_pixel_tumor):
        '''
        https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
        https://medium.com/artificialis/swift-and-simple-calculate-object-distance-with-ease-in-just-few-lines-of-code-38889575bb12

        Distance = (Width x FocalLength) / Pixels
        Width = (Distance x Pixels) / FocalLength
        '''

        ratio_SI_over_Tu = avg_depth_tumor / avg_depth_SI 
        diameter_Tumor_box = self.Rs_real * (diameter_pixel_tumor / pixels_SI) * (ratio_SI_over_Tu)

        return diameter_Tumor_box

    def find_distances_diameters_v2(self, biggest_tumor, Ps1, Ps2, normalized_depth):
        '''
        based on the following link and the papers of proofs
        https://mayavan95.medium.com/3d-position-estimation-of-a-known-object-using-a-single-camera-7a82b37b326b
        '''
        # print(normalized_depth.shape)
        # print(biggest_tumor)
        # print(normalized_depth)
        # normalized_depth = np.array(normalized_depth)
        # print(normalized_depth)
        U_t1, V_t1, U_t2, V_t2 = biggest_tumor
        (U_s1, V_s1) = Ps1
        (U_s2, V_s2) = Ps2
        U_t1 = int(U_t1)
        V_t1 = int(V_t1)
        U_t2 = int(U_t2)
        V_t2 = int(V_t2)
        U_s1 = int(U_s1)
        V_s1 = int(V_s1)
        U_s2 = int(U_s2)
        V_s2 = int(V_s2)

        # tmp = np.zeros_like(normalized_depth)
        # tmp_rgb = cv2.cvtColor(np.uint8(tmp*255), cv2.COLOR_GRAY2BGR)
        # cv2.circle(tmp_rgb,(U_t1, V_t1), radius=0, color=(0, 0, 255), thickness=3)
        # cv2.circle(tmp_rgb,(U_t2, V_t2), radius=0, color=(0, 255, 0), thickness=3)
        # cv2.circle(tmp_rgb,(U_s1, V_s1), radius=0, color=(255, 0, 0), thickness=3)
        # cv2.circle(tmp_rgb,(U_s2, V_s2), radius=0, color=(0, 255, 255), thickness=3)
        # plt.imsave(os.path.join(self.path_save, 'tmp_rgb.png'), tmp_rgb)

        Z_t1 = normalized_depth[V_t1 - 1, U_t1 - 1]
        Z_t2 = normalized_depth[V_t2 - 1, U_t2 - 1]
        Z_s1 = normalized_depth[V_s1 - 1, U_s1 - 1]
        Z_s2 = normalized_depth[V_s2 - 1, U_s2 - 1]

        Z_t1 = 1/Z_t1
        Z_t2 = 1/Z_t2
        Z_s1 = 1/Z_s1
        Z_s2 = 1/Z_s2

        H1 = ((U_s1 - self.args.c_x)/self.args.f_x)**2 + ((V_s1 - self.args.c_y)/self.args.f_y)**2 + 1
        H2 = ((U_s2 - self.args.c_x)/self.args.f_x)**2 + ((V_s2 - self.args.c_y)/self.args.f_y)**2 + 1
        H3 = -2 * (((U_s1 - self.args.c_x)*(U_s2 - self.args.c_x))/(self.args.f_x**2) + ((V_s1 - self.args.c_y)*(V_s2 - self.args.c_y))/(self.args.f_y**2) + 1)

        K1 = ((U_t1 - self.args.c_x)/self.args.f_x)**2 + ((V_t1 - self.args.c_y)/self.args.f_y)**2 + 1
        K2 = ((U_t2 - self.args.c_x)/self.args.f_x)**2 + ((V_t2 - self.args.c_y)/self.args.f_y)**2 + 1
        K3 = -2 * (((U_t1 - self.args.c_x)*(U_t2 - self.args.c_x))/(self.args.f_x**2) + ((V_t1 - self.args.c_y)*(V_t2 - self.args.c_y))/(self.args.f_y**2) + 1)

        M1 = H1 * (Z_s1/Z_s2) + H2 * (Z_s2/Z_s1) + H3
        M2 = K1 * (Z_t1/Z_s1) * (Z_t1/Z_s2) + K2 * (Z_t2/Z_s1) * (Z_t2/Z_s2) + K3 * (Z_t1/Z_s1) * (Z_t2/Z_s2) 

        diameter_Tumor_box = self.Rs_real * np.sqrt((M2/M1))

        return diameter_Tumor_box
    
    def find_distances_diameters_v3(self, object_box, reference_box, normalized_depth):
        '''
        based on the following link and the papers of proofs
        paper: A novel absolute localization estimation of a target with monocular vision
        '''
        # print("reference_box", reference_box)
        # print("object_box", object_box)

        U_obj1, V_obj1, U_obj2, V_obj2 = object_box
        U_ref1, V_ref1, U_ref2, V_ref2 = reference_box

        U_obj1 = int(U_obj1)
        V_obj1 = int(V_obj1)
        U_obj2 = int(U_obj2)
        V_obj2 = int(V_obj2)
        U_ref1 = int(U_ref1)
        V_ref1 = int(V_ref1)
        U_ref2 = int(U_ref2)
        V_ref2 = int(V_ref2)

        top_left_ref = (U_ref1, V_ref1)
        bottom_right_ref = (U_ref2, V_ref2)

        width_px_obj = (U_obj2 - U_obj1)
        hight_px_obj = (V_obj2 - V_obj1)
        S_px_ref = math.sqrt((U_ref2 - U_ref1)**2 + (V_ref2 - V_ref1)**2)
        
        # S_px_obj = width_px_obj * hight_px_obj

        U_Fref = (U_ref1 + U_ref2) / 2
        V_Fref = (V_ref1 + V_ref2) / 2
        U_Fobj = (U_obj1 + U_obj2) / 2
        V_Fobj = (V_obj1 + V_obj2) / 2
        m = ((U_Fref - self.args.c_x) / self.args.f_x)**2 + ((V_Fref - self.args.c_y) / self.args.f_y)**2
        n = ((U_Fobj - self.args.c_x) / self.args.f_x)**2 + ((V_Fobj - self.args.c_y) / self.args.f_y)**2

        d_ref = np.sqrt((self.args.S_real_ref * self.args.f_x * self.args.f_y) / S_px_ref)
        Z_ref = d_ref * np.sqrt(1 + m)

        Depth_obj = normalized_depth[int(V_Fobj) - 1, int(U_Fobj) - 1]
        Depth_ref = normalized_depth[int(V_Fref) - 1, int(U_Fref) - 1]
        Z_obj = Z_ref * (Depth_ref / Depth_obj)

        # Z_obj = self.distance_Midas(d_ref, top_left_ref, bottom_right_ref, normalized_depth, V_Fobj, U_Fobj)

        # angle_obj_ref = math.atan2(abs(V_Fref - V_Fobj), abs(U_Fref - U_Fobj))
        # print("math.sin(angle_obj_ref)", math.sin(angle_obj_ref))
        # Z_obj = Z_ref + (Z_obj - Z_ref) * math.sin(angle_obj_ref)

        H_real_obj = self.args.S_real_ref * (hight_px_obj / S_px_ref) * ((1 + m) / (1 + n)) * ((Z_obj / Z_ref) ** 2)
        W_real_obj = self.args.S_real_ref * (width_px_obj / S_px_ref) * ((1 + m) / (1 + n)) * ((Z_obj / Z_ref) ** 2)
        # S_real_obj = S_real_ref * (S_px_obj / S_px_ref) * ((1+n)/(1+m)) * ((Z_ref/Z_obj)**2)

        # real_W_obj, real_H_obj = self.get_H_W(S_real_obj, width_px_obj, hight_px_obj)     

        return W_real_obj, H_real_obj, Z_ref, Z_obj
    
    def distance_Midas(self, d_ref, top_left_ref, bottom_right_ref, normalized_depth, V_Fobj, U_Fobj):

        (U_tl_ref, V_tl_ref) = top_left_ref
        (U_br_ref, V_br_ref) = bottom_right_ref
        m_tl = ((U_tl_ref - self.args.c_x) / self.args.f_x)**2 + ((V_tl_ref - self.args.c_y) / self.args.f_y)**2
        m_br = ((U_br_ref - self.args.c_x) / self.args.f_x)**2 + ((V_br_ref - self.args.c_y) / self.args.f_y)**2

        Z_tl = d_ref * np.sqrt(1 + m_tl)
        Z_br = d_ref * np.sqrt(1 + m_br)

        # print("Z_tl", Z_tl)
        # print("Z_br", Z_br)

        P_tl = normalized_depth[int(V_tl_ref) - 1, int(U_tl_ref) - 1]
        P_br = normalized_depth[int(V_br_ref) - 1, int(U_br_ref) - 1]
        # print("P_tl", P_tl)
        # print("P_br", P_br)
        scale = (Z_tl - Z_br) / ((Z_br * Z_tl) * (P_br - P_tl))
        shift = (1 / Z_tl) - (scale * P_tl)

        P_obj = normalized_depth[int(V_Fobj) - 1, int(U_Fobj) - 1]
        Z_obj = 1 / (scale * P_obj + shift)

        return Z_obj
    

    def get_H_W(self, S_real_obj, width_px_obj, hight_px_obj):

        ratio_px = width_px_obj / hight_px_obj
        real_H_obj = np.sqrt(S_real_obj / ratio_px)
        real_W_obj = ratio_px * real_H_obj
        return real_W_obj, real_H_obj