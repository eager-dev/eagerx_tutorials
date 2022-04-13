from typing import Optional

import cv2
import numpy as np
import rospy
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, UInt64, Float32MultiArray

from eagerx.core import register as register
from eagerx import process
from eagerx import EngineNode
from eagerx.utils.utils import Msg


class PendulumImage(EngineNode):
    @staticmethod
    @register.spec("PendulumImage", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        process: Optional[int] = process.NEW_PROCESS,
        color: Optional[str] = "cyan",
        shape=[480, 480],
    ):
        """PendulumImage spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(PendulumImage)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["tick", "theta"]
        spec.config.outputs = ["image"]

        # Modify custom node params
        spec.config.shape = shape

        # Set component parameter
        spec.inputs.theta.window = 1

    def initialize(self, shape):
        self.cv_bridge = cv_bridge.CvBridge()
        self.shape = tuple(shape)
        self.render_toggle = False
        self.render_toggle_pub = rospy.Subscriber("%s/env/render/toggle" % self.ns, Bool, self._set_render_toggle)

    @register.states()
    def reset(self):
        # This sensor is stateless (in contrast to e.g. a PID controller).
        pass

    @register.inputs(tick=UInt64, theta=Float32MultiArray)
    @register.outputs(image=Image)
    def callback(self, t_n: float, tick: Msg = None, theta: Float32MultiArray = None):
        state = np.array(theta.msgs[-1].data)
        if self.render_toggle:
            width, height = self.shape
            l = width // 3
            img = np.zeros((height, width, 3), np.uint8)
            sin_theta, cos_theta = np.sin(state[0]), np.cos(state[0])
            img = cv2.line(
                img,
                (width // 2, height // 2),
                (width // 2 + int(l * sin_theta), height // 2 - int(l * cos_theta)),
                (0, 0, 255),
                -1,
            )
            try:
                msg = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")
            except ImportError as e:
                rospy.logwarn_once("[%s] %s. Using numpy instead." % (self.ns_name, e))
                data = img.tobytes("C")
                msg = Image(data=data, height=height, width=width, encoding="bgr8")
        else:
            msg = Image()
        return dict(image=msg)

    def _set_render_toggle(self, msg):
        if msg.data:
            rospy.loginfo("[%s] START RENDERING!" % self.name)
        else:
            rospy.loginfo("[%s] STOPPED RENDERING!" % self.name)
        self.render_toggle = msg.data
