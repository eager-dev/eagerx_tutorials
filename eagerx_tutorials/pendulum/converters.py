# ROS IMPORTS
from std_msgs.msg import Float32

# EAGERX IMPORTS
import eagerx
from eagerx.core.specs import ConverterSpec

# OTHER
import numpy as np
from gym.spaces import Box


class Space_DecomposedAngle(eagerx.SpaceConverter):
    MSG_TYPE_A = np.ndarray
    MSG_TYPE_B = Float32

    @staticmethod
    @eagerx.register.spec("Space_DecomposedAngle", eagerx.SpaceConverter)
    def spec(spec: ConverterSpec, low=None, high=None, dtype="float32"):
        # Initialize spec with default arguments
        spec.initialize(Space_DecomposedAngle)
        params = dict(low=low, high=high, dtype=dtype)
        spec.config.update(params)

    def initialize(self, low=None, high=None, dtype="float32"):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.dtype = dtype

    def get_space(self):
        return Box(self.low, self.high, dtype=self.dtype)

    def A_to_B(self, msg):
        # In this example we only care about going from Float32 to ndarray
        raise NotImplementedError()

    def B_to_A(self, msg):
        return np.array([np.sin(msg.data), np.cos(msg.data)], dtype=self.dtype)
