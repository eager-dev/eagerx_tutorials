import eagerx
from eagerx import register
from typing import Any


class DummyState(eagerx.EngineState):
    @staticmethod
    @register.spec("DummyState", eagerx.EngineState)
    def spec(spec: eagerx.specs.EngineStateSpec):
        spec.initialize(DummyState)

    def initialize(self):
        pass

    def reset(self, state: Any, done: bool):
        pass
