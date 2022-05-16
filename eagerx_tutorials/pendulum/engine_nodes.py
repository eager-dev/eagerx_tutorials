import numpy as np
from typing import Optional

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

    @register.inputs(observation_array=None)
    @register.outputs(observation=None)
    def callback(self, t_n: float, observation_array: Optional[Msg] = None):
        data = np.array(observation_array.msgs[-1][self.idx], dtype="float32")
        return dict(observation=data)
