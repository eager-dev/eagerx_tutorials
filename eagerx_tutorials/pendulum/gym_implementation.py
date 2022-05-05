import eagerx
from eagerx import register
from eagerx.bridges.openai_gym.bridge import GymBridge


# This decorator registers the engine-specific implementation for the entity_id="Pendulum".
@register.bridge("Pendulum", GymBridge)
def gym_bridge(spec: eagerx.specs.ObjectSpec, graph: eagerx.EngineGraph):
    """Engine-specific implementation (GymBridge) of the Pendulum object."""
    # Register openai engine-specific nodes (ObservationSensor, ActionActuator, GymImage)
    import eagerx.bridges.openai_gym  # noqa # pylint: disable=unused-import

    # Register tutorial engine-specific nodes (FloatOutput)
    import eagerx_tutorials.pendulum  # noqa # pylint: disable=unused-import

    # Set engine-specific parameters
    spec.GymBridge.env_id = "Pendulum-v1"

    # Create engine states that implement the registered states
    # Note: The GymBridge implementation unfortunately does not support setting the OpenAI environment state,
    #       nor does it support changing the dynamic parameters.
    #       However, you could create a Bridge specifically for the Pendulum-v1 environment.
    spec.GymBridge.states.model_state = eagerx.EngineState.make("DummyState")  # Use dummy state, so it can still be selected.
    spec.GymBridge.states.model_parameters = eagerx.EngineState.make("DummyState")  # Use dummy state (same reason as above).

    # Create sensor engine nodes.
    image = eagerx.EngineNode.make(
        "GymImage", "image", rate=spec.sensors.image.rate, shape=spec.config.render_shape, process=2
    )
    theta = eagerx.EngineNode.make("FloatOutput", "theta", rate=spec.sensors.theta.rate, idx=0)
    # Create engine node that implements the dtheta observation
    # START EXERCISE 1.1.a
    dtheta = eagerx.EngineNode.make("FloatOutput", "dtheta", rate=spec.sensors.dtheta.rate, idx=1)
    # END EXERCISE 1.1.a

    # Create actuator engine node
    action = eagerx.EngineNode.make("ActionActuator", "action", rate=spec.actuators.u.rate, process=2, zero_action=[0])

    # Use the observations produced by the "Pendulum-v1" to obtain theta and dtheta.
    # Because this observation is [sin(theta), cos(theta), dtheta], so we first convert it to [theta, dtheta]
    x = eagerx.EngineNode.make("ObservationSensor", "x", rate=spec.sensors.theta.rate, process=2)
    x.outputs.observation.converter = eagerx.Processor.make("Angle_DecomposedAngle", convert_to="theta_dtheta")

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
    graph.connect(actuator="u", target=action.inputs.action, converter=eagerx.Processor.make("Negate_Float32MultiArray"))
