#!/usr/bin/env python3

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

image_topic = "/camera/rgb/image_raw"
image_path = "/home/albert/Downloads/bus_image.jpg"


def main():
    rospy.init_node("image_publisher_node")
    pub = rospy.Publisher(image_topic, Image, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(0.1)  # Publish at 0.1 Hz

    # Load the image using OpenCV
    cv_image = cv2.imread(image_path)
    if cv_image is None:
        rospy.logerr(f"Failed to load image from {image_path}")
        return

    while not rospy.is_shutdown():
        # Convert OpenCV image to ROS Image message
        header = Header()
        header.stamp = rospy.Time.now()
        ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8", header=header)
        pub.publish(ros_image)
        rospy.loginfo(f"Published image to {image_topic}")
        rate.sleep()


if __name__ == "__main__":
    main()
