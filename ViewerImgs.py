"""
Reading  image files for the udacity Challenge-two
The images must be available in the directories left, center, right
Require also the files camera.csv and steering.csv
-dataset
   -left
       --*.jpg files
   -center
       --*.jpg files
   -right
       --*.jpg files
   -cameras.csv
   -steering.csv

See the script bagdump.py for info on how to extract from the bag file
available on Github:
https://github.com/rwightman/udacity-driving-reader

See the script https://github.com/commaai/research/blob/master/view_steering_model.py
For the steering angle

"""

### Libraries

import numpy as np
import cv2
import os
from datetime import datetime
from skimage import transform as tf

#Use this variable to filter images which has not much steering movement.
# This value is equal to 30 degree. Set this value to zero, if you are want to use all the angles
#ANGLE_THRESHOLD = 0.523599
ANGLE_THRESHOLD = 0.0

class ViewerImgs():

    image_at_timestamp = []
    angles_at_timestamps = []
    speed_at_timestamps = []

    # ***** get perspective transform for images *****
    rsrc = \
     [[43.45456230828867, 118.00743250075844],
      [104.5055617352614, 69.46865203761757],
      [114.86050156739812, 60.83953551083698],
      [129.74572757609468, 50.48459567870026],
      [132.98164627363735, 46.38576532847949],
      [301.0336906326895, 98.16046448916306],
      [238.25686790036065, 62.56535881619311],
      [227.2547443287154, 56.30924933427718],
      [209.13359962247614, 46.817221154818526],
      [203.9561297064078, 43.5813024572758]]
    rdst = \
     [[10.822125594094452, 1.42189132706374],
      [21.177065426231174, 1.5297552836484982],
      [25.275895776451954, 1.42189132706374],
      [36.062291434927694, 1.6376192402332563],
      [40.376849698318004, 1.42189132706374],
      [11.900765159942026, -2.1376192402332563],
      [22.25570499207874, -2.1376192402332563],
      [26.785991168638553, -2.029755283648498],
      [37.033067044190524, -2.029755283648498],
      [41.67121717733509, -2.029755283648498]]


    def __init__(self, imgDir):

        self.pathDir = imgDir
        self.cameraFile = os.path.join(imgDir, 'camera.csv')
        self.steeringFile = os.path.join(imgDir, 'steering.csv')
	self.tform3_img = tf.ProjectiveTransform()
	self.tform3_img.estimate(np.array(self.rdst), np.array(self.rsrc))

    def readImages(self):
        """
        Read the file camera.csv with the image file name
        """

        #Read the file camera.csv for the image file name
        lines = [line.strip() for line in open(self.cameraFile)]
        i = 0;
	self.centers = []
	self.lefts = []
	self.rights = []

        for line in lines:
            info = line.split(',')
            

            if info[0] == 'seq':
                i += 1
                continue
            
            if info[4] == 'left_camera':
                self.lefts.append(info)
            if info[4] == 'center_camera':
                self.centers.append(info)
            if info[4] == 'right_camera':
                self.rights.append(info)
            i += 1

        print "Total Frames: %d " % (len(self.centers))

    def get_angles_at_timestamps(self, image_timestamps, angle_timestamps, use_average = True):
        angle_idx = 0
	self.angles_at_timestamps = []
	self.speed_at_timestamps = []

    	for image_idx in range(len(image_timestamps)):
            # go through angle values until we reach current image time
            angles_until_timestamps = []
            speed_until_timestamps = []
            while angle_idx < len(angle_timestamps) and \
				angle_timestamps[angle_idx][1] <= image_timestamps[image_idx][1]:
                angles_until_timestamps.append(float(angle_timestamps[angle_idx][2]))
                speed_until_timestamps.append(float(angle_timestamps[angle_idx][4]))
                angle_idx += 1
	    
            avg_angle = np.average(angles_until_timestamps)
	    avg_speed = np.average(speed_until_timestamps)
            # If the speed is zero, skip the image. Also skip if average angle is less that angle threshold
            #Comment the below line, if you need to use all the images for processing
            #if not 0 in speed_until_timestamps and abs(avg_angle) >= ANGLE_THRESHOLD:
	    #Comment the below line if you want to skip speed 0 and avg_angle < ANGLE_THRESHOLD
	    if abs(avg_angle) >= ANGLE_THRESHOLD:
                if use_average:
                    # at any image timestamp, use the average value of the steering from last known timestamp
                    self.angles_at_timestamps.append(avg_angle)
		    self.speed_at_timestamps.append(avg_speed)
                else:
                    # at any image timestamp, use the last known steering angle
                    self.angles_at_timestamps.append(angle_timestamps[angle_idx - 1][2])
		    self.speed_at_timestamps.append(angle_timestamps[angle_idx][4])

    def perspective_tform(self, x, y):
	p1, p2 = self.tform3_img((x,y))[0]
	return p2, p1

    # ***** functions to draw lines *****
    def draw_pt(self, img, x, y, color, sz=1):
	row, col = self.perspective_tform(x, y)
	if row >= 0 and row < img.shape[0] and\
	    col >= 0 and col < img.shape[1]:
	    img[row-sz:row+sz, col-sz:col+sz] = color

    def draw_path(self, img, path_x, path_y, color):
	for x, y in zip(path_x, path_y):
	    self.draw_pt(img, x, y, color)

    # ***** functions to draw predicted path *****

    def calc_curvature(self, v_ego, angle_steers, angle_offset=0):
	deg_to_rad = np.pi/180.
	slip_fator = 0.0014 # slip factor obtained from real data
	steer_ratio = 15.3*0.5  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
	wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

	angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
	curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
	return curvature

    def calc_lookahead_offset(self, v_ego, angle_steers, d_lookahead, angle_offset=0):
	#*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
	curvature = self.calc_curvature(v_ego, angle_steers, angle_offset)

	# clip is to avoid arcsin NaNs due to too sharp turns
	y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
	return y_actual, curvature

    def draw_path_on(self, img, speed_ms, angle_steers, color=(0,255,0)):
        path_x = np.arange(0., 50.1, 0.5)
        path_y, _ = self.calc_lookahead_offset(speed_ms, angle_steers, path_x)
        self.draw_path(img, path_x, path_y, color)

    def readSteering(self):
	"""
        Read the file steering.csv with the image file name

        """

        #Read the file steering.csv with the angles and speed
        lines = [line.strip() for line in open(self.steeringFile)]
        self.steeringLines = []

	for line in lines:
            info = line.split(',')
	    if info[0] == 'seq':
                continue
	    self.steeringLines.append(info)

	self.get_angles_at_timestamps(self.centers, self.steeringLines, use_average = False)
	print len(self.angles_at_timestamps)

    def timestampToStr(self, value):
	timeVal = float(value)/1000000000.0
        dt = datetime.fromtimestamp(timeVal)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    def putTextToImg(self, img, str_Seq, str_Time, height):
	cv2.putText(img, str_Seq, (5,height-5), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, \
                        cv2.LINE_AA)
        cv2.putText(img, str_Time, (70,height-5), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, \
                        cv2.LINE_AA)

    def displayImg(self):
        """
        Display the three images in a frame
        Reduce the image size in 0.5
        The size of the new image will be:
        0.5*(width*3) x 0.5*height
        """

	# If you want to skip n frames, set value to 0 to see all images
	SKIP = 4500
        for idx in range(len(self.centers)):
	    if idx < SKIP:
		continue
            file_left = self.lefts[idx][5]
            file_center = self.centers[idx][5]
            file_right = self.rights[idx][5]

            img_left = cv2.imread(os.path.join(self.pathDir, file_left), \
                                               cv2.IMREAD_COLOR)
            img_center = cv2.imread(os.path.join(self.pathDir, file_center), \
                                                cv2.IMREAD_COLOR)
            img_right = cv2.imread(os.path.join(self.pathDir, file_right), \
                                                cv2.IMREAD_COLOR)

	    #Resize the image to 50%
            img_l = cv2.resize(img_left, None, fx=0.5, fy=0.5, \
                               interpolation = cv2.INTER_LINEAR)
            img_c = cv2.resize(img_center, None, fx=0.5, fy=0.5, \
                               interpolation = cv2.INTER_LINEAR)
            img_r = cv2.resize(img_right, None, fx=0.5, fy=0.5, \
                               interpolation = cv2.INTER_LINEAR)
       
            height, width = img_c.shape[:2]
            new_img = np.zeros((height, width*3, img_c.shape[2]),
                               np.uint8)

            #Adding sequence numbers and Time
	    #Left
            strTime = self.timestampToStr(self.lefts[idx][1])
	    self.putTextToImg(img_l, self.lefts[idx][0], strTime, height)
	    #Center
	    strTime = self.timestampToStr(self.centers[idx][1])
	    self.putTextToImg(img_c, self.centers[idx][0], strTime, height)
	    #Right
	    strTime = self.timestampToStr(self.rights[idx][1])
	    self.putTextToImg(img_r, self.rights[idx][0], strTime, height)
	    
	    angle = float(self.angles_at_timestamps[idx])
	    speed = float(self.speed_at_timestamps[idx])

	    print "speed: %f - angle: %f" % (speed, angle)

	    self.draw_path_on(img_c, speed, angle)

	    #Generate the new image
            for i in range(height):
                new_img[i] = np.concatenate((img_l[i, : ], img_c[i, : ], \
                                             img_r[i, : ]))
                               

            cv2.imshow('Udacity Challenge 2 - Viewer', new_img)
            key = cv2.waitKey(30)

            # Press q to exit
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

            
if __name__ == "__main__":
    app = ViewerImgs('/path_to_Dataset/output/dataset')
    app.readImages()
    app.readSteering()
    app.displayImg()

