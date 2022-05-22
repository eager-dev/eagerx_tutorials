from typing import List

# ROS IMPORTS
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
from math import pi

# EAGERx IMPORTS
from eagerx_ode.engine import OdeEngine
from eagerx import Object, EngineNode, SpaceConverter, EngineState
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Pendulum(Object):
    entity_id = "Pendulum"

    @staticmethod
    @register.sensors(theta=Float32, dtheta=Float32, image=Image, u_applied=Float32MultiArray)
    @register.actuators(u=Float32MultiArray)
    @register.engine_states(
        model_state=Float32MultiArray,
        model_parameters=Float32MultiArray,
        mass=Float32,
        length=Float32,
        max_speed=Float32,
    )
    @register.config(render_shape=[480, 480], render_fn="pendulum_render_fn")
    def agnostic(spec: ObjectSpec, rate: float):
        """Agnostic definition of the Pendulum.

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
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.theta.rate = rate
        spec.sensors.theta.space_converter = SpaceConverter.make("Space_Float32", low=-9999, high=9999, dtype="float32")

        spec.sensors.dtheta.rate = rate
        spec.sensors.dtheta.space_converter = SpaceConverter.make("Space_Float32", low=-9999, high=9999, dtype="float32")

        spec.sensors.image.rate = rate / 2
        spec.sensors.image.space_converter = SpaceConverter.make(
            "Space_Image", low=0, high=255, shape=spec.config.render_shape, dtype="uint8"
        )

        spec.sensors.u_applied.rate = rate
        spec.sensors.u_applied.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-2], high=[2], dtype="float32"
        )

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.u.rate = rate
        spec.actuators.u.window = 1
        spec.actuators.u.space_converter = SpaceConverter.make("Space_Float32MultiArray", low=[-2], high=[2], dtype="float32")

        # Set model_state properties: (space_converters)
        spec.states.model_state.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-pi, -9], high=[pi, 9], dtype="float32"
        )

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
        spec.states.model_parameters.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=low, high=high, dtype="float32"
        )

        mass_low = 0.045
        mass_high = 0.055
        spec.states.mass.space_converter = SpaceConverter.make("Space_Float32", low=mass_low, high=mass_high, dtype="float32")

        length_low = 0.04
        length_high = 0.07
        spec.states.length.space_converter = SpaceConverter.make(
            "Space_Float32", low=length_low, high=length_high, dtype="float32"
        )

        max_speed = 22
        spec.states.max_speed.space_converter = SpaceConverter.make(
            "Space_Float32", low=max_speed, high=max_speed, dtype="float32"
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
        name: str,
        actuators: List[str] = None,
        sensors: List[str] = None,
        states: List[str] = None,
        rate: float = 30.0,
        render_shape: List[int] = None,
        render_fn: str = None,
    ):
        """Object spec of Pendulum"""
        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = ["theta", "dtheta"] if sensors is None else sensors
        spec.config.actuators = ["u"] if actuators is None else actuators
        spec.config.states = ["model_state"] if states is None else states

        # Add custom agnostic params
        spec.config.render_shape = render_shape if render_shape else [480, 480]
        spec.config.render_fn = render_fn if render_fn else "pendulum_render_fn"

        # Add engine implementation
        Pendulum.agnostic(spec, rate)

    @staticmethod
    @register.engine(entity_id, OdeEngine)  # This decorator pre-initializes engine implementation with default object_params
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeEngine) of the object."""
        # Set object arguments
        spec.OdeEngine.ode = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_ode"
        spec.OdeEngine.Dfun = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_dfun"
        # Set default params of pendulum ode [J, m, l, b, K, R, c, d].
        spec.OdeEngine.ode_params = [
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
        spec.OdeEngine.states.model_state = EngineState.make("OdeEngineState")
        spec.OdeEngine.states.model_parameters = EngineState.make("OdeParameters", list(range(8)))
        spec.OdeEngine.states.mass = EngineState.make("DummyState")
        spec.OdeEngine.states.length = EngineState.make("DummyState")
        spec.OdeEngine.states.max_speed = EngineState.make("DummyState")

        # Create sensor engine nodes
        x = EngineNode.make("OdeOutput", "x", rate=spec.sensors.theta.rate, process=2)

        # For didactic purposes, we create two sensors, i.e. one with angle and one with angular velocity.
        # We could also have created a sensor that contains both, but in this way it is more clear which sensor
        # contains what information.
        theta = EngineNode.make("FloatOutput", "theta", rate=spec.sensors.theta.rate, idx=0)
        dtheta = EngineNode.make("FloatOutput", "dtheta", rate=spec.sensors.dtheta.rate, idx=1)

        u_applied = EngineNode.make("ActionApplied", "u_applied", rate=spec.sensors.u_applied.rate, process=2)

        render_fn = f"eagerx_tutorials.pendulum.pendulum_render/{spec.config.render_fn}"
        image = EngineNode.make(
            "OdeRender",
            "image",
            render_fn=render_fn,
            rate=spec.sensors.image.rate,
            process=2,
        )

        # Create actuator engine nodes
        u = EngineNode.make("OdeInput", "u", rate=spec.actuators.u.rate, process=2, default_action=[0])

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
