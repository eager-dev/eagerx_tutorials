import eagerx
from eagerx import register
from eagerx.engines.openai_gym.engine import GymEngine
from eagerx_tutorials.pendulum.objects import Pendulum


# This decorator registers the engine-specific implementation for the entity_id="Pendulum".
@register.engine(GymEngine, entity=Pendulum)
def gym_engine(spec: eagerx.specs.ObjectSpec, graph: eagerx.EngineGraph):
    """Engine-specific implementation (GymEngine) of the Pendulum object."""
    # Import the openai engine-specific nodes (ObservationSensor, ActionActuator, GymImage)
    from eagerx.engines.openai_gym.enginenodes import ObservationSensor, ActionActuator, GymImage

    # Import the tutorial engine-specific nodes (FloatOutput)
    from eagerx_tutorials.pendulum.engine_nodes import FloatOutput

    # Set engine-specific parameters
    spec.engine.env_id = "Pendulum-v1"

    # Create engine states that implement the registered states
    # Note: The GymEngine implementation unfortunately does not support setting the OpenAI environment state,
    #       nor does it support changing the dynamic parameters.
    #       However, you could create an Engine specifically for the Pendulum-v1 environment.
    from eagerx_tutorials.pendulum.engine_states import DummyState, SetGymAttribute

    spec.engine.states.model_state = DummyState.make()  # Use dummy state, so it can still be selected.
    spec.engine.states.model_parameters = DummyState.make()  # Use dummy state (same reason as above).
    spec.engine.states.mass = SetGymAttribute.make(attribute="m")
    spec.engine.states.length = SetGymAttribute.make(attribute="l")
    spec.engine.states.max_speed = SetGymAttribute.make(attribute="max_speed")

    # Create sensor engine nodes.
    image = GymImage.make("image", rate=spec.sensors.image.rate, shape=spec.config.render_shape, process=2)
    theta = FloatOutput.make("theta", rate=spec.sensors.theta.rate, idx=0)
    # Create engine node that implements the dtheta observation
    # START EXERCISE 1.1.a
    dtheta = FloatOutput.make("dtheta", rate=spec.sensors.dtheta.rate, idx=1)
    # END EXERCISE 1.1.a

    # Create actuator engine node
    action = ActionActuator.make("action", rate=spec.actuators.u.rate, process=2, zero_action=[0])

    # Use the observations produced by the "Pendulum-v1" to obtain theta and dtheta.
    # Because this observation is [sin(theta), cos(theta), dtheta], so we first convert it to [theta, dtheta]
    from eagerx_tutorials.pendulum.processor import ObsWithDecomposedAngle

    x = ObservationSensor.make("x", rate=spec.sensors.theta.rate, process=2)
    x.outputs.observation.processor = ObsWithDecomposedAngle.make(convert_to="theta_dtheta")

    # Add all engine nodes to the engine-specific graph
    graph.add([x, theta, image, action])
    # START EXERCISE 1.1.b
    graph.add(dtheta)
    # END EXERCISE 1.1.b

    # theta
    graph.connect(source=x.outputs.observation, target=theta.inputs.observation_array)
    graph.connect(source=theta.outputs.observation, sensor="theta")

    # dtheta
    # START EXERCISE 1.1.c
    graph.connect(source=x.outputs.observation, target=dtheta.inputs.observation_array)
    graph.connect(source=dtheta.outputs.observation, sensor="dtheta")
    # END EXERCISE 1.1.c

    # image
    graph.connect(source=image.outputs.image, sensor="image")

    # u
    # Note: not to be confused with sensor "u", for which we do not provide an implementation here.
    # Note: We add a processor that negates the action, as the torque in OpenAI gym is defined counter-clockwise.
    from eagerx_tutorials.pendulum.processor import VoltageToMotorTorque

    action.inputs.action.processor = VoltageToMotorTorque.make(K=0.03333, R=7.731)
    graph.connect(actuator="u", target=action.inputs.action)
