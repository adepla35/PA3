#!/usr/bin/env python
import rosbag
bag = rosbag.Bag('grid.bag')
for topic, msg, t in bag.read_messages(topics=['Movements', 'Observations']):
    print (msg)
bag.close()