#!/usr/bin/env python

import rospy
import rosbag
import rospkg
import numpy as n
import sys
import math
import os
from std_msgs.msg import Int32, String
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler

landmarks = n.array([[125, 525], [125, 325], [125, 125],[425, 125], [425, 325], [425, 525]])
alpha = 15
beta = 90
threshold = 0.08 
grid = n.zeros((35, 35, 36))
new_grid = n.zeros((35, 35, 36))
pt_count = 0
pi = math.pi
radian_to_degree = 180/math.pi
degree_to_radian = math.pi/180

def point(grid,total_prob):
    grid /= total_prob
    od = n.argmax(grid)
    od_angle = od % grid.shape[2]
    od = od / grid.shape[2]
    od_y = od % grid.shape[1]
    od = od / grid.shape[1]
    od_x = od % grid.shape[0]
    return od_x,od_y,od_angle

def get_pointOnGrid(i, j, k):
	translation_x = i * alpha + getalpha_beta('trans')
	translation_y = j * alpha + getalpha_beta('trans')
	rotation = -180 + k * beta + getalpha_beta('rot')
	return rotation, translation_x, translation_y

def get_obs(i, j, k, tagnum):
    global landmarks
    rotation, translation_x, translation_y = get_point_on_grid(i, j, k)
    trans1_temp = landmarks[tagnum, 0] / 100.0
    trans2_temp = landmarks[tagnum, 1] / 100.0
    translation = n.sqrt((translation_x - trans1_temp) ** 2 + (translation_y - trans2_temp) ** 2)
    tag_angle = (n.arctan2((trans2_temp) - translation_y, (trans1_temp) - translation_x)) * radian_to_degree
    rotation1 = tag_angle - rotation
    return rotation1, translation

def get_alphaBeta(pa):
    if pa == 'rot':
        return beta/2.0
    else:
        return alpha/2.0

def applyGaussian(x, mean, st_deviation):
	val = (1.0 / (((2 * pi)**0.5) * st_deviation)) * n.power(n.e, -1.0 * (((x - mean)**2)/(2.0 * st_deviation ** 2)))
	return val

def main():
    global landmarks
    rate = rospy.Rate(10)
    grid[12, 28, 3] = 1 
    bag = rosbag.Bag('grid.bag')
    for topic, msg, t in bag.read_messages(topics=['Movements', 'Observations']):
        print (msg)
        if topic == 'Movements':
            rot1, rot2, translation = get_mvmnts_from_bag(msg)
            belief = to_position_from_belief(rot1, translation*100.0, rot2)
            normalization = n.sum(belief)
            belief = belief/normalization
            max_belief = n.unravel_index(n.argmax(belief), belief.shape)
            string  = "Robot is at : {} with Probability value : {}".format(str(max_belief),str(n.amax(belief)))
            print(string)
            f.write(string + '\n')
            
        else: 
            tag_num, rng, rotation = get_obs_from_bag(msg)
            b = to_obs(tag_num, rng*100.0, rotation)

    bag.close()

def get_Observations(msg):
    tag_num = msg.tagNum
    rng = msg.range
    rot = msg.bearing
    rotation = radian_to_degree * (euler_from_quaternion(
        [rot.x, rot.y, rot.z, rot.w])[2])
    return tag_num, rng, rotation

def to_obs(tagnum, trans, rot):
    global grid, new_grid, threshold
    new_grid = grid
    grid = n.copy(new_grid)
    total_probability = 0
    for i in range(0, 35):
        for j in range(0, 35):
            for k in range(0, 4):
                rot_tmp, trans_tmp = get_obs(i, j, k, tagnum)
                rot_prb = applyGaussian(rot_tmp, rot, getalpha_beta('rot'))
                trans_prb = applyGaussian(trans_tmp, trans, getalpha_beta('trans'))
                val = new_grid[i, j, k] * trans_prb * rot_prb
                grid[i, j, k] = val
                total_probability += val
    x,y,z = point(grid,total_probability)
    return x,y,z 

def get_Movements(msg):
    translation = msg.translation
    yaw1 =  (euler_from_quaternion(
        [msg.rotation1.x, msg.rotation1.y, msg.rotation1.z, msg.rotation1.w])[2])
    yaw2 = (euler_from_quaternion(
        [msg.rotation2.x, msg.rotation2.y, msg.rotation2.z, msg.rotation2.w])[2])
    rot1 = yaw1 * radian_to_degree
    rot2 = yaw2 * radian_to_degree
    return rot1, rot2, translation

def positionFromBelief(rot1, trans, rot2):
    global grid, new_grid, threshold
    new_grid = grid
    grid = n.copy(new_grid)
    total_prob = 0
    for a in range(0, 35):
        for b in range(0, 35):
            for c in range(0, 4):
                if new_grid[a, b, c] > threshold:
                    total_prob = prob_calc(a, b, c, rot1, trans, rot2, new_grid, grid, total_prob)
    return total_prob

def prob_calc(a, b, c, rotation1, translation, rotation2, new_grid, grid, total_prob):
    for i in range(0, 35):
                for j in range(0, 35):
                        for k in range(0, 4):
                            rot1_1, trans1_x, trans1_y = get_point_on_grid(i, j, k)
                            rot2_1, trans2_x, trans2_y = get_point_on_grid(a, b, c)
                            delta_translation = n.sqrt((trans1_x - trans2_x) ** 2 + (trans1_y - trans2_y) ** 2)
                            delta_rotation1 = (n.arctan2(trans1_y-trans2_y, trans1_x - trans2_x)) * radian_to_degree
                            rot2 = rot2_1 - rot1_1 - delta_rotation1
                            rot1 = delta_rotation1
                            rotation1_tmp, delta_translation_tmp, rotation2_tmp = rot1, delta_translation, rot2
                            rotation1_prob = applyGaussian(rotation1_tmp, rotation1, getalpha_beta('rot'))
                            translation_prob = applyGaussian(delta_translation_tmp, translation, getalpha_beta('trans'))
                            rotation2_prob = applyGaussian(rotation2_tmp, rotation2, getalpha_beta('rot'))
                            curr_grid = new_grid[a, b, c] * translation_prob * rotation1_prob * rotation2_prob
                            grid[i, j, k] += curr_grid
                            total_prob += grid
    return total_prob

if __name__ == '__main__':
    if not rospy.is_shutdown():
        try:
            rospy.init_node('Bayes_Filter')
            main()
        except rospy.ROSInterruptException as e:
            print('Exception Occured :: {}'.format(str(e)))