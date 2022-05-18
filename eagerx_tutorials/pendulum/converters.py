# ROS IMPORTS
from std_msgs.msg import Float32, Float32MultiArray

# EAGERX IMPORTS
import eagerx

# OTHER
import numpy as np
from gym.spaces import Box


class Space_DecomposedAngle(eagerx.SpaceConverter):
    MSG_TYPE_A = np.ndarray
    MSG_TYPE_B = Float32

    @staticmethod
    @eagerx.register.spec("Space_DecomposedAngle", eagerx.SpaceConverter)
    def spec(spec: eagerx.specs.ConverterSpec, low: float, high: float, dtype: str = "float32"):
        spec.config.update(low=low, high=high, dtype=dtype)

    def initialize(self, low: float, high: float, dtype: str = "float32"):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.dtype = dtype

    def get_space(self):
        return Box(self.low, self.high, dtype=self.dtype)

    def A_to_B(self, msg):
        # In this example we only care about going from Float32 to ndarray
        raise NotImplementedError()

    def B_to_A(self, msg):
        return np.array([np.cos(msg.data), np.sin(msg.data)], dtype=self.dtype)


class Angle_DecomposedAngle(eagerx.Processor):
    MSG_TYPE = Float32MultiArray

    @staticmethod
    @eagerx.register.spec("Angle_DecomposedAngle", eagerx.Processor)
    def spec(spec: eagerx.specs.ConverterSpec, convert_to: str = "theta_dtheta"):
        spec.config.update(convert_to=convert_to)

    def initialize(self, convert_to: str):
        self.convert_to = convert_to

    def convert(self, msg: Float32MultiArray) -> Float32MultiArray:
        if not len(msg.data):  # No data
            return msg
        elif self.convert_to == "trig_dtheta":
            data = [np.sin(-msg.data[0]), np.cos(msg.data[0]), -msg.data[1]]
        elif self.convert_to == "theta_dtheta":
            cos_th = msg.data[0]
            sin_th = msg.data[1]
            data = [-np.arctan2(sin_th, cos_th), -msg.data[2]]
        else:
            raise NotImplementedError(f"Convert_to '{self.convert_to}' not implemented.")
        return Float32MultiArray(data=data)


class Negate_Float32MultiArray(eagerx.Processor):
    MSG_TYPE = Float32MultiArray

    @staticmethod
    @eagerx.register.spec("Negate_Float32MultiArray", eagerx.Processor)
    def spec(spec: eagerx.specs.ConverterSpec):
        pass

    def initialize(self):
        pass

    def convert(self, msg: Float32MultiArray) -> Float32MultiArray:
        return Float32MultiArray(data=[-i for i in msg.data])


class Voltage_MotorTorque(eagerx.Processor):
    MSG_TYPE = Float32MultiArray

    @staticmethod
    @eagerx.register.spec("Voltage_MotorTorque", eagerx.Processor)
    def spec(spec: eagerx.specs.ConverterSpec, K: float, R: float):
        # Initialize spec with default arguments
        spec.initialize(Voltage_MotorTorque)

        spec.config.K = K
        spec.config.R = R

    def initialize(self, K: float, R: float):
        self.K = K
        self.R = R

    def convert(self, msg: Float32MultiArray) -> Float32MultiArray:
        return Float32MultiArray(data=[-msg.data[0] * self.K / self.R])
