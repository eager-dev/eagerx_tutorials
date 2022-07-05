from typing import Optional
import numpy as np
from gym.spaces import Box

# IMPORT EAGERX
from eagerx.core.constants import process
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register


class FloatOutput(EngineNode):
    @classmethod
    def make(
        cls,
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
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["observation_array"]
        spec.config.outputs = ["observation"]

        # Custom node params
        spec.config.idx = idx
        return spec

    def initialize(self, spec, object_spec, simulator):
        self.obj_name = object_spec.config.name
        self.idx = spec.config.idx

    @register.states()
    def reset(self):
        pass

    @register.inputs(observation_array=None)
    @register.outputs(observation=Box(low=-9999, high=9999, shape=(), dtype="float32"))
    def callback(self, t_n: float, observation_array: Optional[Msg] = None):
        data = observation_array.msgs[-1].data
        return dict(observation=np.array(data[self.idx], dtype="float32"))
