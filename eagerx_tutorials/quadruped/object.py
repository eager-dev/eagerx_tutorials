import os
from typing import List, Optional

import eagerx
import eagerx.core.register as register
import numpy as np
from eagerx import EngineNode, EngineState, Object, SpaceConverter
from eagerx.core.graph_engine import EngineGraph
from eagerx.core.specs import ObjectSpec
from eagerx_pybullet.engine import PybulletEngine
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

import eagerx_tutorials.quadruped.cartesian_control  # noqa: F401
import eagerx_tutorials.quadruped.go1.configs_go1 as go1_config


class Quadruped(Object):
    entity_id = "Quadruped"

    @staticmethod
    @register.sensors(
        joint_position=Float32MultiArray,
        joint_velocity=Float32MultiArray,
        force_torque=Float32MultiArray,
        orientation=Float32MultiArray,
        position=Float32MultiArray,
        velocity=Float32MultiArray,
        image=Image,
    )
    @register.actuators(
        joint_control=Float32MultiArray,
    )
    @register.engine_states(
        joint_position=Float32MultiArray,
        position=Float32MultiArray,
        orientation=Float32MultiArray,
        velocity=Float32MultiArray,
        angular_velocity=Float32MultiArray,
    )
    @register.config(
        joint_names=None,
        fixed_base=False,
        self_collision=True,
        position=None,
        orientation=None,
        control_mode=None,
        render_shape=None,
    )
    def agnostic(spec: ObjectSpec, rate):
        """This methods builds the agnostic definition for a quadruped.

        Registered (agnostic) config parameters (should probably be set in the spec() function):
        - joint_names: List of quadruped joints.
        - fixed_base: Force the base of the loaded object to be static.
        - self_collision: Enable self collisions.
        - base_pos: Base position of the object [x, y, z].
        - base_orientation: Base orientation of the object in quaternion [x, y, z, w].
        - control_mode: Control mode for the arm joints.
                        Available: `position_control`, `velocity_control`, `pd_control`, and `torque_control`.

        :param spec: Holds the desired configuration.
        :param rate: Rate (Hz) at which the callback is called.
        """
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa

        # Set observation properties: (space_converters, rate, etc...)
        # TODO: specify correct limits
        spec.sensors.joint_position.rate = rate
        spec.sensors.joint_position.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.RL_LOWER_ANGLE_JOINT.tolist(),
            high=go1_config.RL_UPPER_ANGLE_JOINT.tolist(),
        )

        # TODO: specify correct limits
        spec.sensors.joint_velocity.rate = rate
        spec.sensors.joint_velocity.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=(-go1_config.VELOCITY_LIMITS).tolist(),
            high=go1_config.VELOCITY_LIMITS.tolist(),
        )

        # TODO: specify correct limits
        # TODO: HIGH DIMENSIONAL!! 6 measurements / joint
        spec.sensors.force_torque.rate = rate
        spec.sensors.force_torque.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[0] * 6 * len(go1_config.RL_LOWER_ANGLE_JOINT),
            high=[0] * 6 * len(go1_config.RL_LOWER_ANGLE_JOINT),
        )

        # TODO: specify correct limits
        spec.sensors.orientation.rate = rate
        spec.sensors.orientation.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=list(go1_config.INIT_ORIENTATION),
            high=list(go1_config.INIT_ORIENTATION),
        )

        # TODO: specify correct limits
        spec.sensors.position.rate = rate
        spec.sensors.position.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-10.0, -10.0, 0.0],
            high=[10.0, 10.0, 0.5],
        )

        spec.sensors.velocity.rate = rate
        spec.sensors.velocity.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-1.0, -1.0, -0.2],
            high=[1.0, 1.0, 0.2],
        )

        # Rgb
        spec.sensors.image.rate = rate
        spec.sensors.image.space_converter = SpaceConverter.make(
            "Space_Image",
            dtype="float32",
            low=0,
            high=1,
            shape=spec.config.render_shape + [3],
        )

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.joint_control.rate = rate
        spec.actuators.joint_control.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.RL_LOWER_ANGLE_JOINT.tolist(),
            high=go1_config.RL_UPPER_ANGLE_JOINT.tolist(),
        )

        # Set model_state properties: (space_converters)
        spec.states.joint_position.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.INIT_JOINT_ANGLES.tolist(),
            high=go1_config.INIT_JOINT_ANGLES.tolist(),
        )

        spec.states.position.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.INIT_POSITION,
            high=go1_config.INIT_POSITION,
        )

        spec.states.orientation.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=list(go1_config.INIT_ORIENTATION),
            high=list(go1_config.INIT_ORIENTATION),
        )

        spec.states.velocity.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[0, 0, 0],
            high=[0, 0, 0],
        )

        spec.states.angular_velocity.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[0, 0, 0],
            high=[0, 0, 0],
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
        name: str,
        sensors: Optional[List[str]] = None,
        actuators: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        rate: float = 30.0,
        position: Optional[List[int]] = None,
        orientation: Optional[List[int]] = None,
        self_collision: bool = False,
        fixed_base: bool = False,
        control_mode: str = "position_control",
        render_shape: Optional[List[int]] = None,
    ):
        """A spec to create a go1 robot.

        :param spec: The desired object configuration.
        :param name: Name of the object (topics are placed within this namespace).
        :param sensors: A list of selected sensors. Must be a subset of the registered sensors.
        :param actuators: A list of selected actuators. Must be a subset of the registered actuators.
        :param states: A list of selected states. Must be a subset of the registered actuators.
        :param rate: The default rate at which all sensors and actuators run. Can be modified via the spec API.
        :param position: Base position of the object [x, y, z].
        :param orientation: Base orientation of the object in quaternion [x, y, z, w].
        :param self_collision: Enable self collisions.
        :param fixed_base: Force the base of the loaded object to be static.
        :param control_mode: Control mode for the arm joints. Available: `position_control`, `velocity_control`, `pd_control`, and `torque_control`.
        :param render_shape: Render shape (height, width) of the image produced by the image sensor.
        :return: ObjectSpec
        """
        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["joint_position", "joint_velocity", "orientation"]
        spec.config.actuators = actuators if actuators else ["joint_control"]
        spec.config.states = (
            states if states else ["joint_position", "position", "orientation", "velocity", "angular_velocity"]
        )

        # Add registered agnostic params
        spec.config.joint_names = list(go1_config.JOINT_NAMES)
        spec.config.position = position if position else go1_config.INIT_POSITION
        spec.config.orientation = orientation if orientation else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base
        spec.config.control_mode = control_mode
        spec.config.render_shape = render_shape if isinstance(render_shape, list) else [400, 400]

        # Add agnostic implementation
        Quadruped.agnostic(spec, rate)

    @staticmethod
    # This decorator pre-initializes engine implementation with default object_params
    @register.engine(entity_id, PybulletEngine)
    def pybullet_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Import any object specific entities for this engine
        import eagerx_pybullet  # noqa

        # Set object arguments (as registered per register.engine_params(..) above the engine.add_object(...) method.
        urdf_file = os.path.join(go1_config.URDF_ROOT, go1_config.URDF_FILENAME)
        spec.PybulletEngine.urdf = urdf_file
        spec.PybulletEngine.basePosition = spec.config.position
        spec.PybulletEngine.baseOrientation = spec.config.orientation
        spec.PybulletEngine.fixed_base = spec.config.fixed_base
        spec.PybulletEngine.self_collision = spec.config.self_collision

        # Create engine_states (no agnostic states defined in this case)
        spec.PybulletEngine.states.joint_position = EngineState.make(
            "JointState", joints=spec.config.joint_names, mode="position"
        )

        spec.PybulletEngine.states.position = EngineState.make("LinkState", mode="position")
        spec.PybulletEngine.states.orientation = EngineState.make("LinkState", mode="orientation")
        spec.PybulletEngine.states.velocity = EngineState.make("LinkState", mode="velocity")
        spec.PybulletEngine.states.angular_velocity = EngineState.make("LinkState", mode="angular_vel")

        # Create sensor engine nodes
        # Rate=None, but we will connect them to sensors (thus will use the rate set in the agnostic specification)
        rate = spec.sensors.joint_position.rate
        joint_position = EngineNode.make(
            "JointSensor",
            "joint_position",
            rate=rate,
            process=eagerx.process.ENGINE,
            joints=spec.config.joint_names,
            mode="position",
        )

        joint_velocity = EngineNode.make(
            "JointSensor",
            "joint_velocity",
            rate=rate,
            process=eagerx.process.ENGINE,
            joints=spec.config.joint_names,
            mode="velocity",
        )
        force_torque = EngineNode.make(
            "JointSensor",
            "force_torque",
            rate=rate,
            process=eagerx.process.ENGINE,
            joints=spec.config.joint_names,
            mode="force_torque",
        )

        # TODO: convert to euler (currently quaternion)
        orientation = EngineNode.make(
            "LinkSensor",
            "orientation",
            rate=rate,
            process=eagerx.process.ENGINE,
            links=None,
            mode="orientation",
        )
        position = EngineNode.make(
            "LinkSensor",
            "position",
            rate=rate,
            process=eagerx.process.ENGINE,
            links=None,
            mode="position",
        )
        velocity = EngineNode.make(
            "LinkSensor",
            "velocity",
            rate=rate,
            process=eagerx.process.ENGINE,
            links=None,
            mode="velocity",
        )
        image = EngineNode.make(
            "CameraSensor",
            "image",
            rate=spec.sensors.image.rate,
            process=eagerx.process.ENGINE,
            mode="rgb",
            render_shape=spec.config.render_shape,
        )
        image.config.fov = 80.0  # todo: tune

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        joint_control = EngineNode.make(
            "JointController",
            "joint_control",
            rate=spec.actuators.joint_control.rate,
            process=eagerx.process.ENGINE,
            joints=spec.config.joint_names,
            mode=spec.config.control_mode,
            vel_target=np.zeros(len(go1_config.JOINT_NAMES)).tolist(),
            pos_gain=np.ones(len(go1_config.JOINT_NAMES)).tolist(),
            vel_gain=np.ones(len(go1_config.JOINT_NAMES)).tolist(),
            max_force=(10 * np.ones(len(go1_config.JOINT_NAMES))).tolist(),
        )
        # Connect all engine nodes
        graph.add(
            [
                joint_position,
                joint_velocity,
                force_torque,
                orientation,
                position,
                velocity,
                joint_control,
                image,
            ]
        )
        graph.connect(source=joint_position.outputs.obs, sensor="joint_position")
        graph.connect(source=joint_velocity.outputs.obs, sensor="joint_velocity")
        graph.connect(source=force_torque.outputs.obs, sensor="force_torque")
        graph.connect(source=orientation.outputs.obs, sensor="orientation")
        graph.connect(source=position.outputs.obs, sensor="position")
        graph.connect(source=velocity.outputs.obs, sensor="velocity")
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(actuator="joint_control", target=joint_control.inputs.action)

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
