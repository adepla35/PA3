#!/usr/bin/env python
import rosbag
import yaml
import tf
import math
import numpy as np
from scipy.stats import mvn
from scipy.stats import norm
import matplotlib.mlab as mlab
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

degreeQuantum = math.pi/2.0
pos = [12, 28]
heading = 200.52
noise = 0.1
gridQuantum = 0.2
numCells = 35
prevGridProb = np.zeros((4,35,35))
path = []
landmarks = {0:{'x': 1.25, 'y': 5.25}, 1:{'x': 1.25,'y':3.25}, 2:{'x': 1.25,'y':1.25}, 3:{'x': 4.25,'y':1.25}, 4:{'x': 4.25,'y':3.25}, 5:{'x': 4.25,'y':5.25}}
angleProb = {0: 0.68268949213708585, 1: 0.15730535589982697, 2: 0.0013496113800581799, 3: 0.15730535589982697}

def publish(index, pub, i):
    marker = Marker()
    marker.header.frame_id = "marker"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "marker"
    marker.id = i
    marker.action = Marker.ADD
    marker.type = Marker.POINTS
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.color.b = 1.0
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration(0)
    point = Point()
    point.x = index[0]
    point.y = index[1]
    print (index)
    marker.points.append(point)
    pub.publish(marker)  

def quaternionToEuler(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    (roll,pitch,yaw) = tf.transformations.euler_from_quaternion((x, y, z, w))
    return math.degrees(yaw)

def cos(deg):
    return math.cos(math.radians(deg))

def sin(deg):
    return math.sin(math.radians(deg))

def rad(angle):
    return math.radians(angle*degreeQuantum)

def deg(rad):
    return math.degrees(rad)

def print_NonZero(gridProb):
    nz = np.nonzero(gridProb)
    ind = zip(nz[0], nz[1], nz[2])
    for i in ind:
        print (i,'-->',gridProb[i[0]][i[1]][i[2]])

def mvnpdf(x, y, mux, muy, sigma):
    X = np.copy(x)
    Y = np.copy(y)
    X -= mux
    Y -= muy
    X = X**2
    Y = Y**2
    XY = (X+Y)/(2*(sigma**2))
    z = np.exp(-XY)
    z /= z.sum()
    return z

def updateHeading(rot, gridProb):
    print ('rot:',quaternionToEuler(rot))
    for x in range(35):
        for y in range(35):
            temp = np.zeros(4)
            for h in range(4):
                p = gridProb[h][y][x]
                newHeadingMean = ((90*h) + quaternionToEuler(rot))%360
                for posterior in range(4):
                    angleBin = np.floor(abs(posterior*90 - newHeadingMean)/90)
                    temp[angleBin] += p*angleProb[posterior]
            gridProb[:, y, x] = temp
    gridProb /= gridProb.sum()

def updatePosition(trans, gridProb):
    print ('trabs:',trans)
    tempGrid = np.zeros((4,35,35))
    X,Y = np.meshgrid(range(35), range(35))
    X = X*0.2 + 0.1
    Y = Y*0.2 + 0.1    
    for h in range(4):
        for x in np.array(range(35)):
            for y in np.array(range(35)):
                xp = x*0.2 + 0.1
                yp = y*0.2 + 0.1
                if h == 0:
                    mean = (xp + trans, yp)
                elif h == 1:
                    mean = (xp, yp + trans)
                elif h == 2:
                    mean = (xp - trans, yp)
                else:
                    mean = (xp, yp - trans)
                p = gridProb[h][y][x]
                gridProb[h][y][x] = 0.0
                Z = mvnpdf(X, Y, mean[0], mean[1], 0.1)
                Z *= p
                tempGrid[h] += Z
    for direction in range(4):
        gridProb[direction, :, :] = tempGrid[direction, :, :]                
    gridProb /= gridProb.sum()
       
def handle_Movements(message, gridProb):
    updateHeadingBelief(message.rotation1, gridProb)
    print ('after rot1:',np.where(np.logical_and(gridProb == gridProb.max(), gridProb.max() > 0)))
    updatePositionBelief(message.translation, gridProb)
    print ('after trans:',np.where(np.logical_and(gridProb == gridProb.max(), gridProb.max() > 0)))
    updateHeadingBelief(message.rotation2, gridProb)
    print ('after rot2:',np.where(np.logical_and(gridProb == gridProb.max(), gridProb.max() > 0)))

def handle_Observations(message, gridProb):
    X,Y = np.meshgrid(range(35), range(35))
    X = X*0.2 + 0.1
    Y = Y*0.2 + 0.1
    mean = landmarks[message.tagNum]
    Z = mvnpdf(X, Y, mean['x'], mean['y'], 0.1)
    for h in range(4):
        theta = h*90+quaternionToEuler(message.bearing)
        deltaLandmark = (message.range*cos(theta), message.range*sin(theta))
        for x in range(35):
            for y in range(35):
                expectedLandmark = (np.floor((np.array((x*0.2, y*0.2))+deltaLandmark)/0.2))
                if max(expectedLandmark) < 35:
                    gridProb[h][y][x] *= Z[expectedLandmark[1]][expectedLandmark[0]]
                else:
                    gridProb[h][y][x] *= 0
    gridProb /= gridProb.sum()
    print ('after obs:',np.where(np.logical_and(gridProb == gridProb.max(), gridProb.max() > 0)))

def readROSbag(filename):
    bag = rosbag.Bag('grid.bag')
    for topic, msg, t in bag.read_messages(topics=['Movements', 'Observations']):
        print (msg)
    bag.close()           

if __name__ == '__main__':
    #rospy.init_node('bayes_filter')
    np.set_printoptions(threshold=5000, precision=4)
    readROSbag('grid.bag')
    