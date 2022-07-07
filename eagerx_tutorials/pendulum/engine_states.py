import eagerx
from eagerx.core.specs import EngineStateSpec, ObjectSpec
from typing import Any


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls) -> EngineStateSpec:
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, object_spec: ObjectSpec, simulator: Any):
        pass

    def reset(self, state: Any):
        pass


class SetGymAttribute(eagerx.EngineState):
    @classmethod
    def make(cls, attribute: str) -> EngineStateSpec:
        spec = cls.get_specification()
        spec.config.attribute = attribute
        return spec

    def initialize(self, spec: EngineStateSpec, object_spec: ObjectSpec, simulator: Any):
        self.attribute = spec.config.attribute
        self.simulator = simulator

    def reset(self, state: Any):
        for _obj_name, sim in self.simulator.items():
            if hasattr(sim["env"].env, self.attribute):
                setattr(sim["env"].env, self.attribute, state)
            else:
                self.backend.logwarn_once(f"{self.attribute} is not an attribute of the environment.")
