from typing import Optional

# IMPORT ROS
from std_msgs.msg import Float32MultiArray, Float32

# IMPORT EAGERX
from eagerx.core.constants import process
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register


class FloatOutput(EngineNode):
    @staticmethod
    @register.spec("FloatOutput", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        idx: Optional[int] = 0,
        process: Optional[int] = process.ENVIRONMENT,
        color: Optional[str] = "cyan",
    ):
        """
        FloatOutput spec

        :param idx: index of the value of interest from the array.
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(FloatOutput)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["observation_array"]
        spec.config.outputs = ["observation"]

        # Custom node params
        spec.config.idx = idx

    def initialize(self, idx):
        self.obj_name = self.config["name"]
        self.idx = idx

    @register.states()
    def reset(self):
        pass

    @register.inputs(observation_array=Float32MultiArray)
    @register.outputs(observation=Float32)
    def callback(self, t_n: float, observation_array: Optional[Msg] = None):
        data = observation_array.msgs[-1].data[self.idx]
        return dict(observation=Float32(data=data))
