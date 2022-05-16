from typing import List

# ROS IMPORTS
from gym.spaces import Box
import numpy as np
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
from math import pi

# EAGERx IMPORTS
from eagerx_ode.engine import OdeEngine
from eagerx import Object, EngineNode, EngineState
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Pendulum(Object):
    entity_id = "Pendulum"

    @staticmethod
    @register.sensors(theta=Box(low=-np.inf, high=np.inf, shape=(), dtype="float32"),
                      dtheta=Box(low=-np.inf, high=np.inf, shape=(), dtype="float32"),
                      image=None,
                      u_applied=Box(low=-3, high=3, shape=(1,), dtype="float32"))
    @register.actuators(u=Box(low=-3, high=3, shape=(1,), dtype="float32"))
    @register.engine_states(model_state=None,
                            model_parameters=None)
    @register.config(render_shape=[480, 480])
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
        model_parameters: allows resetting ODE parameters [J, m, l, b, K, R]

        Config
        render_shape: shape of render window [height, width]
        """
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Set rates
        spec.sensors.theta.rate = rate
        spec.sensors.dtheta.rate = rate
        spec.sensors.image.rate = 15
        spec.sensors.u_applied.rate = rate
        spec.actuators.u.rate = rate

        # Set not-yet defined spaces
        shape = (spec.config.render_shape[0], spec.config.render_shape[1], 3)
        spec.sensors.image.space = Box(low=0, high=255, shape=shape, dtype="uint8")
        spec.states.model_state.space = Box(low=np.array([-pi, -9], dtype="float32"), high=np.array([pi, 9], dtype="float32"))
        mean = [0.0002, 0.05, 0.04, 0.0001, 0.05, 9.0]
        diff = [0.05, 0, 0, 0.05, 0.05, 0.05]  # Percentual delta with respect to fixed value
        low = np.array([val - diff * val for val, diff in zip(mean, diff)], dtype="float32")
        high = np.array([val + diff * val for val, diff in zip(mean, diff)], dtype="float32")
        spec.states.model_parameters.space = Box(low=low, high=high)

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
        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = ["theta", "dtheta"] if sensors is None else sensors
        spec.config.actuators = ["u"] if actuators is None else actuators
        spec.config.states = ["model_state"] if states is None else states

        # Add custom agnostic params
        spec.config.render_shape = render_shape if render_shape else [480, 480]

        # Add engine implementation
        Pendulum.agnostic(spec, rate)

    @staticmethod
    @register.engine(entity_id, OdeEngine)  # This decorator pre-initializes engine implementation with default object_params
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeEngine) of the object."""
        # Set object arguments
        spec.OdeEngine.ode = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_ode"
        spec.OdeEngine.Dfun = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_dfun"
        # Set default params of pendulum ode [J, m, l, b, K, R].
        spec.OdeEngine.ode_params = [0.0002, 0.05, 0.04, 0.0001, 0.05, 9.0]

        # Create engine_states (no agnostic states defined in this case)
        spec.OdeEngine.states.model_state = EngineState.make("OdeEngineState")

        # Create engine_states (no agnostic states defined in this case)
        spec.OdeEngine.states.model_parameters = EngineState.make("OdeParameters", list(range(5)))

        # Create sensor engine nodes
        x = EngineNode.make("OdeOutput", "x", rate=spec.sensors.theta.rate, process=2)

        # For didactic purposes, we create two sensors, i.e. one with angle and one with angular velocity.
        # We could also have created a sensor that contains both, but in this way it is more clear which sensor
        # contains what information.
        theta = EngineNode.make("FloatOutput", "theta", rate=spec.sensors.theta.rate, idx=0)
        dtheta = EngineNode.make("FloatOutput", "dtheta", rate=spec.sensors.dtheta.rate, idx=1)

        u_applied = EngineNode.make("ActionApplied", "u_applied", rate=spec.sensors.u_applied.rate, process=2)

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
        graph.add([x, theta, dtheta, image, action, u_applied])

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
        graph.connect(source=action.outputs.action_applied, target=u_applied.inputs.action_applied, skip=True)
        graph.connect(source=u_applied.outputs.action_applied, sensor="u_applied")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
