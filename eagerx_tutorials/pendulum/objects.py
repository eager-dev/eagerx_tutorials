from typing import List

# ROS IMPORTS
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
from math import pi

# EAGERx IMPORTS
from eagerx_ode.bridge import OdeBridge
from eagerx import Object, EngineNode, SpaceConverter, EngineState
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Pendulum(Object):
    entity_id = "Pendulum"

    @staticmethod
    @register.sensors(theta=Float32, dtheta=Float32, image=Image, u=Float32MultiArray)
    @register.actuators(u=Float32MultiArray)
    @register.engine_states(model_state=Float32MultiArray, model_parameters=Float32MultiArray)
    @register.config(render_shape=[480, 480])
    def agnostic(spec: ObjectSpec, rate: float):
        """Agnostic definition of the Pendulum.

        Sensors
        theta: angle of the pendulum wrt upward position
        dtheta: angular velocity of the pendulum
        image: render of pendulum system
        u: DC motor voltage

        Actuators
        u: DC motor voltage

        States
        model_state: allows resetting the angle and angular velocity
        model_parameters: allows resetting ODE parameters [J, m, l, b, K, R]

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

        spec.sensors.image.rate = 15
        spec.sensors.image.space_converter = SpaceConverter.make(
            "Space_Image", low=0, high=255, shape=spec.config.render_shape, dtype="uint8"
        )

        spec.sensors.u.rate = rate
        spec.sensors.u.space_converter = SpaceConverter.make("Space_Float32MultiArray", low=[-3], high=[3], dtype="float32")

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.u.rate = rate
        spec.actuators.u.window = 1
        spec.actuators.u.space_converter = SpaceConverter.make("Space_Float32MultiArray", low=[-3], high=[3], dtype="float32")

        # Set model_state properties: (space_converters)
        spec.states.model_state.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-pi, -9], high=[pi, 9], dtype="float32"
        )

        # Set model_parameters properties: (space_converters)
        mean = [0.0002, 0.05, 0.04, 0.0001, 0.05, 9.0]
        diff = [0.05, 0, 0, 0.05, 0.05, 0.05]  # Percentual delta with respect to fixed value
        low = [val - diff * val for val, diff in zip(mean, diff)]
        high = [val + diff * val for val, diff in zip(mean, diff)]
        spec.states.model_parameters.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=low, high=high, dtype="float32"
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
    ):
        """Object spec of Pendulum"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        Pendulum.initialize_spec(spec)

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = ["theta", "dtheta"] if sensors is None else sensors
        spec.config.actuators = ["u"] if actuators is None else actuators
        spec.config.states = ["model_state"] if states is None else states

        # Add custom agnostic params
        spec.config.render_shape = render_shape if render_shape else [480, 480]

        # Add bridge implementation
        Pendulum.agnostic(spec, rate)

    @staticmethod
    @register.bridge(entity_id, OdeBridge)  # This decorator pre-initializes bridge implementation with default object_params
    def ode_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeBridge) of the object."""
        # Set object arguments (nothing to set here in this case)
        spec.OdeBridge.ode = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_ode"
        spec.OdeBridge.Dfun = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_dfun"
        # Set default params of pendulum ode [J, m, l, b, K, R].
        spec.OdeBridge.ode_params = [0.0002, 0.05, 0.04, 0.0001, 0.05, 9.0]

        # Create engine_states (no agnostic states defined in this case)
        spec.OdeBridge.states.model_state = EngineState.make("OdeEngineState")

        # Create engine_states (no agnostic states defined in this case)
        spec.OdeBridge.states.model_parameters = EngineState.make("OdeParameters", list(range(5)))

        # Create sensor engine nodes
        x = EngineNode.make("OdeOutput", "x", rate=spec.sensors.theta.rate, process=2)

        # For didactic purposes, we create two sensors, i.e. one with angle and one with angular velocity.
        # We could also have created a sensor that contains both, but in this way it is more clear which sensor
        # contains what information.
        theta = EngineNode.make("FloatOutput", "theta", rate=spec.sensors.theta.rate, idx=0)
        dtheta = EngineNode.make("FloatOutput", "dtheta", rate=spec.sensors.dtheta.rate, idx=1)

        u = EngineNode.make("ActionApplied", "u", rate=spec.sensors.u.rate, process=2)

        image = EngineNode.make(
            "OdeRender",
            "image",
            render_fn="eagerx_tutorials.pendulum.pendulum_render/pendulum_render_fn",
            rate=spec.sensors.image.rate,
            process=2,
        )

        # Create actuator engine nodes
        action = EngineNode.make("OdeInput", "pendulum_actuator", rate=spec.actuators.u.rate, process=2, default_action=[0])

        # Connect all engine nodes
        graph.add([x, theta, dtheta, image, action, u])

        # theta
        graph.connect(source=x.outputs.observation, target=theta.inputs.observation_array)
        graph.connect(source=theta.outputs.observation, sensor="theta")

        # dtheta
        graph.connect(source=x.outputs.observation, target=dtheta.inputs.observation_array)
        graph.connect(source=dtheta.outputs.observation, sensor="dtheta")

        # image
        graph.connect(source=x.outputs.observation, target=image.inputs.observation)
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(source=action.outputs.action_applied, target=image.inputs.action_applied, skip=True)

        # u
        graph.connect(actuator="u", target=action.inputs.action)
        graph.connect(source=action.outputs.action_applied, target=u.inputs.action_applied, skip=True)
        graph.connect(source=u.outputs.action_applied, sensor="u")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
