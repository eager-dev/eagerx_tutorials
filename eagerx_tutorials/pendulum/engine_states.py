import rospy
import eagerx
from eagerx import register
from typing import Any
from std_msgs.msg import Float32


class DummyState(eagerx.EngineState):
    @staticmethod
    @register.spec("DummyState", eagerx.EngineState)
    def spec(spec: eagerx.specs.EngineStateSpec):
        pass

    def initialize(self):
        pass

    def reset(self, state: Any, done: bool):
        pass


class SetGymAttribute(eagerx.EngineState):
    @staticmethod
    @register.spec("SetGymAttribute", eagerx.EngineState)
    def spec(spec: eagerx.specs.EngineStateSpec, attribute: str):
        spec.initialize(SetGymAttribute)
        spec.config.attribute = attribute

    def initialize(self, attribute: str):
        self.attribute = attribute

    def reset(self, state: Float32, done: bool):
        for _obj_name, sim in self.simulator.items():
            if hasattr(sim["env"].env, self.attribute):
                setattr(sim["env"].env, self.attribute, state.data)
            else:
                rospy.logwarn_once(f"{self.attribute} is not an attribute of the environment.")
