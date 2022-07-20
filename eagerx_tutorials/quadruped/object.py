import os
from typing import List, Optional

import eagerx
from eagerx import Space
import eagerx.core.register as register
from eagerx.core.graph_engine import EngineGraph
from eagerx.core.specs import ObjectSpec
from eagerx_pybullet.engine import PybulletEngine

import eagerx_tutorials.quadruped.go1.configs_go1 as go1_config


class Quadruped(eagerx.Object):
    @classmethod
    @register.sensors(
        joint_position=Space(low=go1_config.RL_LOWER_ANGLE_JOINT, high=go1_config.RL_UPPER_ANGLE_JOINT, dtype="float32"),
        joint_velocity=Space(low=-go1_config.VELOCITY_LIMITS, high=go1_config.VELOCITY_LIMITS, dtype="float32"),
        force_torque=Space(  # todo: set realistic bounds for forces and moments.
            low=[-200.0] * 6 * len(go1_config.RL_LOWER_ANGLE_JOINT),
            high=[200.0] * 6 * len(go1_config.RL_LOWER_ANGLE_JOINT),
            dtype="float32",
        ),
        orientation=Space(low=go1_config.INIT_ORIENTATION, high=go1_config.INIT_ORIENTATION, dtype="float32"),
        position=Space(low=[-10.0, -10.0, 0.0], high=[10.0, 10.0, 0.5], dtype="float32"),
        velocity=Space(low=[-1.0, -1.0, -0.2], high=[1.0, 1.0, 0.2], dtype="float32"),
        image=Space(dtype="uint8"),
    )
    @register.actuators(
        joint_control=Space(low=go1_config.RL_LOWER_ANGLE_JOINT, high=go1_config.RL_UPPER_ANGLE_JOINT),
    )
    @register.engine_states(
        joint_position=Space(low=go1_config.INIT_JOINT_ANGLES, high=go1_config.INIT_JOINT_ANGLES, dtype="float32"),
        position=Space(low=go1_config.INIT_POSITION, high=go1_config.INIT_POSITION, dtype="float32"),
        orientation=Space(low=go1_config.INIT_ORIENTATION, high=go1_config.INIT_ORIENTATION, dtype="float32"),
        velocity=Space(low=[0.0, 0.0, 0.0], high=[0.0, 0.0, 0.0], dtype="float32"),
        angular_velocity=Space(low=[0.0, 0.0, 0.0], high=[0.0, 0.0, 0.0], dtype="float32"),
    )
    def make(
        cls,
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
    ) -> ObjectSpec:
        """Make a spec to create a go1 quadruped robot.

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
        spec = cls.get_specification()

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

        # Set rates
        spec.sensors.joint_position.rate = rate
        spec.sensors.joint_velocity.rate = rate
        spec.sensors.force_torque.rate = rate
        spec.sensors.orientation.rate = rate
        spec.sensors.position.rate = rate
        spec.sensors.velocity.rate = rate
        spec.sensors.image.rate = rate
        spec.actuators.joint_control.rate = rate

        # Set variable space limits
        spec.sensors.image.space = Space(
            dtype="uint8",
            low=0,
            high=255,
            shape=tuple(spec.config.render_shape + [3]),
        )

        return spec

    @staticmethod
    @register.engine(PybulletEngine)
    def pybullet_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Set object arguments (as registered per register.engine_params(..) above the engine.add_object(...) method.
        urdf_file = os.path.join(go1_config.URDF_ROOT, go1_config.URDF_FILENAME)
        spec.engine.urdf = urdf_file
        spec.engine.basePosition = spec.config.position
        spec.engine.baseOrientation = spec.config.orientation
        spec.engine.fixed_base = spec.config.fixed_base
        spec.engine.self_collision = spec.config.self_collision

        # Create engine_states (no agnostic states defined in this case)
        from eagerx_pybullet.enginestates import JointState, LinkState

        spec.engine.states.joint_position = JointState.make(joints=spec.config.joint_names, mode="position")

        spec.engine.states.position = LinkState.make(mode="position")
        spec.engine.states.orientation = LinkState.make(mode="orientation")
        spec.engine.states.velocity = LinkState.make(mode="velocity")
        spec.engine.states.angular_velocity = LinkState.make(mode="angular_vel")

        # Create sensor engine nodes
        # Rate=None, but we will connect them to sensors (thus will use the rate set in the agnostic specification)
        from eagerx_pybullet.enginenodes import JointSensor, LinkSensor, CameraSensor, JointController

        rate = spec.sensors.joint_position.rate
        joint_position = JointSensor.make(
            "joint_position",
            rate=rate,
            process=eagerx.process.ENGINE,
            joints=spec.config.joint_names,
            mode="position",
        )

        joint_velocity = JointSensor.make(
            "joint_velocity",
            rate=rate,
            process=eagerx.process.ENGINE,
            joints=spec.config.joint_names,
            mode="velocity",
        )
        force_torque = JointSensor.make(
            "force_torque",
            rate=rate,
            process=eagerx.process.ENGINE,
            joints=spec.config.joint_names,
            mode="force_torque",
        )

        # TODO: convert to euler (currently quaternion)
        orientation = LinkSensor.make(
            "orientation",
            rate=rate,
            process=eagerx.process.ENGINE,
            links=None,
            mode="orientation",
        )
        position = LinkSensor.make(
            "position",
            rate=rate,
            process=eagerx.process.ENGINE,
            links=None,
            mode="position",
        )
        velocity = LinkSensor.make(
            "velocity",
            rate=rate,
            process=eagerx.process.ENGINE,
            links=None,
            mode="velocity",
        )
        image = CameraSensor.make(
            "image",
            rate=spec.sensors.image.rate,
            process=eagerx.process.ENGINE,
            mode="rgb",
            render_shape=spec.config.render_shape,
        )
        image.config.fov = 80.0  # todo: tune

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        joint_control = JointController.make(
            "joint_control",
            rate=spec.actuators.joint_control.rate,
            process=eagerx.process.ENGINE,
            joints=spec.config.joint_names,
            mode=spec.config.control_mode,
            vel_target=[0] * len(go1_config.JOINT_NAMES),
            pos_gain=[1] * len(go1_config.JOINT_NAMES),
            vel_gain=[1] * len(go1_config.JOINT_NAMES),
            max_force=[10] * len(go1_config.JOINT_NAMES),
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
