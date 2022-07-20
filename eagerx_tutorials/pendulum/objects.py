from typing import List

# EAGERx IMPORTS
from eagerx_ode.engine import OdeEngine
from eagerx import Object, Space
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Pendulum(Object):
    @classmethod
    @register.sensors(
        theta=Space(low=-999.0, high=999.0, shape=(), dtype="float32"),
        dtheta=Space(low=-999.0, high=999.0, shape=(), dtype="float32"),
        image=Space(dtype="uint8"),  # shape, low & high determined at run-time
        u_applied=Space(low=[-2.0], high=[2.0], dtype="float32"),
    )
    @register.actuators(u=Space(low=[-2], high=[2], dtype="float32"))
    @register.engine_states(
        model_state=Space(low=[-3.14, -9], high=[3.14, 9], dtype="float32"),
        model_parameters=Space(dtype="float32"),  # shape, low & high determined at run-time
        mass=Space(low=0.01, high=0.07, shape=(), dtype="float32"),
        length=Space(low=0.04, high=0.2, shape=(), dtype="float32"),
        max_speed=Space(low=22, high=22, shape=(), dtype="float32"),
    )
    def make(
        cls,
        name: str,
        actuators: List[str] = None,
        sensors: List[str] = None,
        states: List[str] = None,
        rate: float = 30.0,
        render_shape: List[int] = None,
        render_fn: str = None,
    ):
        """Make the specification of the Pendulum.

        Sensors
        theta: angle of the pendulum wrt upward position
        dtheta: angular velocity of the pendulum
        image: render of pendulum system
        u_applied: Applied DC motor voltage

        Actuators
        u: DC motor voltage

        States
        model_state: allows resetting the angle and angular velocity
        model_parameters: allows resetting all ODE parameters [J, m, l, b, K, R, c, d].
        mass: allows resetting the mass of the Gym pendulum m
        length: allows resetting the length of the Gym pendulum l
        max_speed: allows resetting the max speed of the Gym pendulum

        Config
        render_shape: shape of render window [height, width]
        """
        spec = cls.get_specification()

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = ["theta", "dtheta"] if sensors is None else sensors
        spec.config.actuators = ["u"] if actuators is None else actuators
        spec.config.states = ["model_state"] if states is None else states

        # Add custom agnostic params
        spec.config.render_shape = render_shape if render_shape else [480, 480]
        spec.config.render_fn = render_fn if render_fn else "pendulum_render_fn"

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.theta.rate = rate
        spec.sensors.dtheta.rate = rate
        spec.sensors.image.rate = rate / 2
        spec.sensors.u_applied.rate = rate
        spec.actuators.u.rate = rate

        # Set image space
        shape = (spec.config.render_shape[0], spec.config.render_shape[1], 3)
        spec.sensors.image.space = Space(low=0, high=255, shape=shape, dtype="uint8")

        # Set model_parameters properties: (space_converters)
        # Set default params of pendulum ode [J, m, l, b, K, R, c, d].
        mean = [
            0.000159931461600856,
            0.0508581731919534,
            0.0415233722862552,
            1.43298488358436e-05,
            0.0333391179016334,
            7.73125142447252,
            0.000975041213361349,
            165.417960777425,
        ]
        diff = [0, 0, 0, 0, 0, 0, 0, 0]  # Percentual delta with respect to fixed value
        low = [val - diff * val for val, diff in zip(mean, diff)]
        high = [val + diff * val for val, diff in zip(mean, diff)]
        spec.states.model_parameters.space = Space(low=low, high=high, dtype="float32")

        return spec

    @staticmethod
    @register.engine(OdeEngine)  # This decorator pre-initializes engine implementation with default object_params
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeEngine) of the object."""
        # Set object arguments
        spec.engine.ode = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_ode"
        spec.engine.Dfun = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_dfun"

        # Set default params of pendulum ode [J, m, l, b, K, R, c, d].
        spec.engine.ode_params = [
            0.000159931461600856,
            0.0508581731919534,
            0.0415233722862552,
            1.43298488358436e-05,
            0.0333391179016334,
            7.73125142447252,
            0.000975041213361349,
            165.417960777425,
        ]

        # Create engine_states (no agnostic states defined in this case)
        from eagerx_ode.engine_states import OdeEngineState, OdeParameters
        from eagerx_tutorials.pendulum.engine_states import DummyState
        from eagerx_tutorials.pendulum.engine_nodes import FloatOutput
        from eagerx_ode.engine_nodes import OdeOutput, OdeInput, OdeRender, ActionApplied

        spec.engine.states.model_state = OdeEngineState.make()
        spec.engine.states.model_parameters = OdeParameters.make(list(range(8)))
        spec.engine.states.mass = DummyState.make()
        spec.engine.states.length = DummyState.make()
        spec.engine.states.max_speed = DummyState.make()

        # Create sensor engine nodes
        x = OdeOutput.make("x", rate=spec.sensors.theta.rate, process=2)

        # For didactic purposes, we create two sensors, i.e. one with angle and one with angular velocity.
        # We could also have created a sensor that contains both, but in this way it is more clear which sensor
        # contains what information.
        theta = FloatOutput.make("theta", rate=spec.sensors.theta.rate, idx=0)
        dtheta = FloatOutput.make("dtheta", rate=spec.sensors.dtheta.rate, idx=1)

        u_applied = ActionApplied.make("u_applied", rate=spec.sensors.u_applied.rate, process=2)

        render_fn = f"eagerx_tutorials.pendulum.pendulum_render/{spec.config.render_fn}"
        shape = spec.sensors.image.space.shape[:2]
        image = OdeRender.make("image", render_fn=render_fn, rate=spec.sensors.image.rate, process=2, shape=shape)

        # Create actuator engine nodes
        u = OdeInput.make("u", rate=spec.actuators.u.rate, process=2, default_action=[0])

        # Connect all engine nodes
        graph.add([x, theta, dtheta, image, u, u_applied])

        # theta
        graph.connect(source=x.outputs.observation, target=theta.inputs.observation_array)
        graph.connect(source=theta.outputs.observation, sensor="theta")

        # dtheta
        graph.connect(source=x.outputs.observation, target=dtheta.inputs.observation_array)
        graph.connect(source=dtheta.outputs.observation, sensor="dtheta")

        # image
        graph.connect(source=x.outputs.observation, target=image.inputs.observation)
        graph.connect(source=image.outputs.image, sensor="image")

        # u
        graph.connect(actuator="u", target=u.inputs.action)

        graph.connect(source=u.outputs.action_applied, target=image.inputs.action_applied, skip=True)
        graph.connect(source=u.outputs.action_applied, target=u_applied.inputs.action_applied, skip=True)
        graph.connect(source=u_applied.outputs.action_applied, sensor="u_applied")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
