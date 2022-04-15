# ROS IMPORTS
from std_msgs.msg import Float32MultiArray

# RX IMPORTS
import eagerx.core.register as register
from eagerx import SpaceConverter
from eagerx.core.specs import ConverterSpec
import numpy as np
from gym.spaces import Box


class Space_AngleDecomposition(SpaceConverter):
    MSG_TYPE_A = np.ndarray
    MSG_TYPE_B = Float32MultiArray

    @staticmethod
    @register.spec("Space_AngleDecomposition", SpaceConverter)
    def spec(spec: ConverterSpec, low=None, high=None, dtype="float32"):
        # Initialize spec with default arguments
        spec.initialize(Space_AngleDecomposition)
        params = dict(low=low, high=high, dtype=dtype)
        spec.config.update(params)

    def initialize(self, low=None, high=None, dtype="float32"):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.dtype = dtype

    def get_space(self):
        return Box(self.low, self.high, dtype=self.dtype)

    def A_to_B(self, msg):
        return Float32MultiArray(data=msg)

    def B_to_A(self, msg):
        angle = msg.data[0]
        return np.concatenate(([np.sin(angle), np.cos(angle)], msg.data[1:]), axis=0)
