# ROS IMPORTS
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

# EAGERx IMPORTS
from eagerx_ode.bridge import OdeBridge
from eagerx import Object, EngineNode, SpaceConverter, EngineState
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Pendulum(Object):
    entity_id = "Pendulum"

    @staticmethod
    @register.sensors(angle_sensor=Float32MultiArray, action_applied=Float32MultiArray, image=Image)
    @register.actuators(voltage=Float32MultiArray)
    @register.engine_states(model_state=Float32MultiArray, model_parameters=Float32MultiArray)
    @register.config(render_shape=[480, 480])
    def agnostic(spec: ObjectSpec, rate):
        """Agnostic definition of the Pendulum"""
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.angle_sensor.rate = rate
        spec.sensors.angle_sensor.space_converter = SpaceConverter.make(
            "Space_AngleDecomposition", low=[-1, -1, -9], high=[1, 1, 9], dtype="float32"
        )

        spec.sensors.action_applied.rate = rate
        spec.sensors.action_applied.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3], high=[3], dtype="float32"
        )

        spec.sensors.image.rate = 15
        spec.sensors.image.space_converter = SpaceConverter.make(
            "Space_Image", low=0, high=1, shape=spec.config.render_shape, dtype="float32"
        )

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.voltage.rate = rate
        spec.actuators.voltage.window = 1
        spec.actuators.voltage.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3], high=[3], dtype="float32"
        )

        # Set model_state properties: (space_converters)
        spec.states.model_state.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3.14159265359, -9], high=[3.14159265359, 9], dtype="float32"
        )

        # Set model_parameters properties: (space_converters) # [J, m, l, b, K, R]
        fixed = [0.00018, 0.056, 0.044, 0.00014, 0.05, 9.8]
        diff = [0, 0, 0, 0.08, 0.08, 0.08]  # Percentual delta with respect to fixed value
        low = [val - diff * val for val, diff in zip(fixed, diff)]
        high = [val + diff * val for val, diff in zip(fixed, diff)]
        spec.states.model_parameters.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=low, high=high, dtype="float32"
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
        name: str,
        actuators=None,
        sensors=None,
        states=None,
        rate=30,
        render_shape=None,
    ):
        """Object spec of Pendulum"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        Pendulum.initialize_spec(spec)

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["angle_sensor", "action_applied", "image"]
        spec.config.actuators = actuators if actuators else ["voltage"]
        spec.config.states = states if states else ["model_state"]

        # Add registered agnostic params
        spec.config.render_shape = render_shape if render_shape else [480, 480]

        # Add bridge implementation
        Pendulum.agnostic(spec, rate)

    @staticmethod
    @register.bridge(entity_id, OdeBridge)  # This decorator pre-initializes bridge implementation with default object_params
    def ode_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeBridge) of the object."""
        # Import any object specific entities for this bridge
        import eagerx_tutorials.pendulum  # noqa # pylint: disable=unused-import

        # Set object arguments (nothing to set here in this case)
        spec.OdeBridge.ode = "eagerx_tutorials.pendulum.pendulum_ode/pendulum_ode"
        # Set default params of pendulum ode [J, m, l, b, K, R].
        spec.OdeBridge.ode_params = [0.000189238, 0.0563641, 0.0437891, 0.000142205, 0.0502769, 9.83536]

        # Create engine_states (no agnostic states defined in this case)
        spec.OdeBridge.states.model_state = EngineState.make("OdeEngineState")
        spec.OdeBridge.states.model_parameters = EngineState.make("OdeParameters", list(range(6)))

        # Create sensor engine nodes
        obs = EngineNode.make("OdeOutput", "angle_sensor", rate=spec.sensors.angle_sensor.rate, process=2)
        image = EngineNode.make(
            "OdeRender",
            "image",
            shape=spec.config.render_shape,
            render_fn="eagerx_tutorials.pendulum.pendulum_render/pendulum_render_fn",
            rate=spec.sensors.image.rate,
            process=2,
        )

        # Create actuator engine nodes
        action = EngineNode.make(
            "OdeInput", "pendulum_actuator", rate=spec.actuators.voltage.rate, process=2, default_action=[0]
        )

        # Connect all engine nodes
        graph.add([obs, image, action])
        graph.connect(source=obs.outputs.observation, sensor="angle_sensor")
        graph.connect(source=obs.outputs.observation, target=image.inputs.observation)
        graph.connect(source=action.outputs.action_applied, target=image.inputs.action_applied)
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(actuator="voltage", target=action.inputs.action)

        # Add action applied
        applied = EngineNode.make("ActionApplied", "applied", rate=spec.sensors.action_applied.rate, process=2)
        graph.add(applied)
        graph.connect(source=action.outputs.action_applied, target=applied.inputs.action_applied, skip=True)
        graph.connect(source=applied.outputs.action_applied, sensor="action_applied")

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
