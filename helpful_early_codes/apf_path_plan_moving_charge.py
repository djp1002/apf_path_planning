#!/usr/bin/python3
# Demo Trackbar 
# importing cv2 and numpy 
import cv2 
import math
import numpy as np


def nothing(x): 
	pass
def force_cal(p1,p2,charge):
	dist = [p2[0] - p1[0],p2[1] - p1[1],0]
	dir = math.atan2(dist[1],dist[0])
	dist[2] = max((dist[0]**2 + dist[1]**2)**0.5,6)
	
	force = [0,0,0]
	force[2] = 100000*charge/dist[2]**2
	force = [force[2]*math.cos(dir),force[2]*math.sin(dir),force[2]]
	return force

def force_endpoint(p1,f):
	return(p1[0] + int(f[0]), p1[1] + int(f[1]))


def force_resultant(f1,f2):
	f = [(f1[0]+f2[0]),(f1[1]+f2[1]),0]
	f[2] = (f[0]**2 + f[1]**2)**0.5
	return f
def force_unit(f):
	f_unit = [f[0]/f[2],f[1]/f[2]]
	f_min = max(abs(f_unit[0]),abs(f_unit[1]))
	f_unit = [mag*f_unit[0]/f_min,mag*f_unit[1]/f_min]
	return f_unit

def dist_between_points(p1,p2):
	d = [p2[0] - p1[0],p2[1] - p1[1],0]
	d[2] = (d[0]**2 + d[1]**2)**0.5
	return d


def apf(travel_p,goal_p,obs_p,magnitude):
	cell_shift = [0,0] 
	shift_magnitude = 5
	f_goal = force_cal(travel_p,goal_p,charge_goal*(1+ magnitude/100))
	# f_goal_endpoint = force_endpoint(start_p,f_goal)
	# magnitude = min(magnitude,99)
	f_ob = force_cal(travel_p,obs_p,charge_ob*(1-(min(magnitude,99))/100))
	# f_ob_endpoint = force_endpoint(travel_p,f_ob)
	
	f_final = force_resultant(f_goal,f_ob)
	# f_final_endpoint = force_endpoint(travel_p,f_final )
	f_dir = math.atan2(f_final[1],f_final[0])

	cell_shift[0] = math.ceil(shift_magnitude*math.cos(f_dir))
	cell_shift[1] = math.ceil(shift_magnitude*math.sin(f_dir))


	d_to_goal = dist_between_points(travel_p,goal_p)
	# # print(force_final,force_final_unit)
	if (abs(d_to_goal[2])>goal_achieved_dist):
		travel_p = [travel_p[0]+cell_shift[0],travel_p[1]+cell_shift[1] ]
	return travel_p

def cv_window():
	global image
	# image = np.zeros(shape =(480,640,3))
	rect = [[ob_p[0]-rect_dim[0],ob_p[1]-rect_dim[1]],[ob_p[0]+rect_dim[0],ob_p[1]+rect_dim[1]]]
	cv2.rectangle(image,pt1=(rect[0][0],-rect[0][1]),pt2=(rect[1][0],-rect[1][1]),color=(0,0,255),thickness=-1)
	cv2.circle(image,center=(start_p[0],-start_p[1]),radius=20,color=(50,50,0),thickness=1)
	cv2.circle(image,center=(end_p[0],-end_p[1]),radius=20,color=(0,255,0),thickness=-1)
	# cv2.arrowedLine(image,pt1=(start_p[0],-start_p[1]),pt2=(force_ob_endpoint[0],-force_ob_endpoint[1]),color=(0,0,255),thickness=2)
	# cv2.arrowedLine(image,pt1=(start_p[0],-start_p[1]),pt2=(force_goal_endpoint[0],-force_goal_endpoint[1]),color=(0,255,0),thickness=2)
	# cv2.arrowedLine(image,pt1=(start_p[0],-start_p[1]),pt2=(force_final_endpoint[0],-force_final_endpoint[1]),color=(0,255,255),thickness=2)
	# cv2.line(image, (50, 50), (250, 250), color=(255, 0, 0), thickness=5)
	rect_dim[0] = cv2.getTrackbarPos('pt1x', 'value')
	rect_dim[1] = cv2.getTrackbarPos('pt1y', 'value')
	# mag = cv2.getTrackbarPos('size', 'value')
	pass

# def point_on_rectangle(ext_point, rectangle):
#     return [max(rectangle[0][0], min(ext_point[0], rectangle[1][0])),-max(-rectangle[0][1], min(-ext_point[1], -rectangle[1][1]))]

def point_on_rectangle(ext_point, rectangle):
    return [max(rectangle[0][0], min(ext_point[0], rectangle[1][0])),-max(-rectangle[0][1], min(-ext_point[1], -rectangle[1][1]))]

global image
goal_achieved_dist = 20
charge_ob = -2
charge_goal = 10
start_p = [40,-440]
temp_p = start_p
end_p = [300,-140]
ob_p = [175,-175]
rect_dim = [20,100]

# rect = [[150,-150],[200,-200]]
mag = 0
mag_increment = 1
mag_diff_path = 10
previous_path_length = 400
current_path_length = 300
c = 0
image = np.zeros(shape =(480,640,3))

cv2.namedWindow('value') 
cv2.createTrackbar('pt1x', 'value', rect_dim[0], 640,nothing) 
cv2.createTrackbar('pt1y', 'value', rect_dim[1], 480, nothing) 
cv2.createTrackbar('size', 'value', 1, 15, nothing)
rect = [[ob_p[0]-rect_dim[0],ob_p[1]+rect_dim[1]],[ob_p[0]+rect_dim[0],ob_p[1]-rect_dim[1]]]

temp1_p = start_p
temp2_p = start_p

path_1_length = 0
path_2_length = 0
path_3_length = 0

path_1 = []
path_p = []
previous_rect = rect
while(True): 
	image = np.zeros(shape =(480,640,3))
	goal_achieved = [0,0,0]

	temp1_p = start_p
	temp2_p = start_p
	temp3_p = start_p
	path_1 = []
	path_2 = []
	path_3 = []
	# if (current_path_length <= previous_path_length and (charge_ob + mag/100)<0):
	if (path_2_length>path_1_length or (mag == 0)):
		# print(mag,current_path_length,previous_path_length)
		# while(dist_between_points(temp1_p,end_p)[2]>goal_achieved_dist):
		while(not goal_achieved[0] or not goal_achieved[1] or not goal_achieved[2]):
			
			rect = [[ob_p[0]-rect_dim[0],ob_p[1]+rect_dim[1]],[ob_p[0]+rect_dim[0],ob_p[1]-rect_dim[1]]]

			if(dist_between_points(temp1_p,end_p)[2]>goal_achieved_dist):
				# rect = [[ob_p[0]-rect_dim[0],ob_p[1]+rect_dim[1]],[ob_p[0]+rect_dim[0],ob_p[1]-rect_dim[1]]]
				point_rectangle = point_on_rectangle(temp1_p,rect)
				temp1_p = apf(temp1_p,end_p,point_rectangle,magnitude=mag+(mag_increment*mag_diff_path))
				path_1.append(temp1_p)
				if (len(path_1)>2):
					cv2.line(image,(path_1[-2][0], -path_1[-2][1]),(path_1[-1][0], -path_1[-1][1]),color=(255,0,0),thickness=1)
					cv2.circle(image,center=(point_rectangle[0],-point_rectangle[1]),radius=1,color=(0,255,0),thickness=-1)
			else:
				goal_achieved[0] = 1
			if(dist_between_points(temp2_p,end_p)[2]>goal_achieved_dist):
				# rect = [[ob_p[0]-rect_dim[0],ob_p[1]+rect_dim[1]],[ob_p[0]+rect_dim[0],ob_p[1]-rect_dim[1]]]
				point_rectangle = point_on_rectangle(temp2_p,rect)
				temp2_p = apf(temp2_p,end_p,point_rectangle,magnitude=mag)
				path_2.append(temp2_p)
				if (len(path_2)>2):
					cv2.line(image,(path_2[-2][0], -path_2[-2][1]),(path_2[-1][0], -path_2[-1][1]),color=(255,0,0),thickness=1)
					cv2.circle(image,center=(point_rectangle[0],-point_rectangle[1]),radius=1,color=(0,255,0),thickness=-1)
			else:
				goal_achieved[1] = 1
			if(dist_between_points(temp3_p,end_p)[2]>goal_achieved_dist):
				# rect = [[ob_p[0]-rect_dim[0],ob_p[1]+rect_dim[1]],[ob_p[0]+rect_dim[0],ob_p[1]-rect_dim[1]]]
				point_rectangle = point_on_rectangle(temp3_p,rect)
				temp3_p = apf(temp3_p,end_p,point_rectangle,magnitude=mag-(mag_increment*mag_diff_path))
				path_3.append(temp3_p)
				if (len(path_3)>2):
					cv2.line(image,(path_3[-2][0], -path_3[-2][1]),(path_3[-1][0], -path_3[-1][1]),color=(255,0,0),thickness=1)
					cv2.circle(image,center=(point_rectangle[0],-point_rectangle[1]),radius=1,color=(0,255,0),thickness=-1)
			else:
				goal_achieved[2] = 1
		previous_path_length = current_path_length
		current_path_length = len(path_2)
		path_1_length = len(path_1)
		path_2_length = len(path_2)
		path_3_length = len(path_3)
		previous_rect = rect
		path_p = path_2
		# print(mag,current_path_length,previous_path_length,charge_ob + mag/100,charge_goal + mag/100)
		mag += mag_diff_path
		# time.sleep(0.01)
		print(len(path_1),len(path_2),len(path_3),mag)
	else:
		print("optimum length",len(path_p),"magnitude",mag)
		# time.sleep(0.05)
		for i in range(len(path_p)-2):
			rect = [[ob_p[0]-rect_dim[0],ob_p[1]+rect_dim[1]],[ob_p[0]+rect_dim[0],ob_p[1]-rect_dim[1]]]
			cv2.line(image,(path_p[i][0], -path_p[i][1]),(path_p[i+1][0], -path_p[i+1][1]),color=(255,255,0),thickness=1)
			point_rectangle = point_on_rectangle(path_p[i],rect)
			cv2.circle(image,center=(point_rectangle[0],-point_rectangle[1]),radius=1,color=(0,255,0),thickness=-1)
			

	cv_window()
	cv2.imshow('value', image)	
	# for button pressing and changing 
	k = cv2.waitKey(1) & 0xFF
	if k == 27: 
		break
	# time.sleep(0.05)
# close the window 
cv2.destroyAllWindows() 