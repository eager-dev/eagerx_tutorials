import eagerx
from eagerx.core.specs import EngineStateSpec
from typing import Any


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls) -> EngineStateSpec:
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        pass

    def reset(self, state: Any):
        pass


class SetGymAttribute(eagerx.EngineState):
    @classmethod
    def make(cls, attribute: str) -> EngineStateSpec:
        spec = cls.get_specification()
        spec.config.attribute = attribute
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        self.attribute = spec.config.attribute
        self.simulator = simulator

    def reset(self, state: Any):
        attr_set = False
        env = self.simulator["env"]
        while hasattr(env, "env"):
            env = env.env
            if hasattr(env, self.attribute):
                setattr(env, self.attribute, state)
                attr_set = True
        if not attr_set:
            self.backend.logwarn_once(f"{self.attribute} is not an attribute of the environment.")
