from typing import Optional

import eagerx
import numpy as np
from eagerx import register
from eagerx.utils.utils import Msg
from std_msgs.msg import Float32MultiArray

import eagerx_tutorials.quadruped.go1.configs_go1 as go1_config


class CartesiandPDController(eagerx.Node):
    @staticmethod
    @register.spec("CartesiandPDController", eagerx.Node)
    def spec(
        spec: eagerx.specs.NodeSpec,
        name: str,
        rate: float,
        process: Optional[int] = eagerx.process.ENVIRONMENT,
    ):
        """A spec to create a CartesiandPDController node that controls a set of joints.

        :param spec: Holds the desired configuration in a Spec object.
        :param name: User specified node name.
        :param rate: Rate (Hz) at which the callback is called.
        :param process: Process in which this node is launched. See :class:`~eagerx.core.constants.process` for all options.
        :return: NodeSpec
        """
        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process)
        spec.config.update(inputs=["cartesian_pos"], outputs=["joint_pos"], joints=list(go1_config.JOINT_NAMES))

        # TODO: fix correct limits
        spec.inputs.cartesian_pos.space_converter = eagerx.SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.NOMINAL_FOOT_POS_LEG_FRAME.tolist(),
            high=go1_config.NOMINAL_FOOT_POS_LEG_FRAME.tolist(),
        )

    def initialize(self, joints):
        self.joints = joints

    @register.states()
    def reset(self):
        pass

    @register.inputs(cartesian_pos=Float32MultiArray)
    @register.outputs(joint_pos=Float32MultiArray)
    def callback(self, t_n: float, cartesian_pos: Msg):
        # desired [x, y, z] for each joint
        action = np.array(cartesian_pos.msgs[-1].data)
        desired_joint_angles = np.zeros(len(self.joints))
        # call inverse kinematics to get corresponding joint angles
        for leg_idx in range(go1_config.NUM_LEGS):
            xyz_desired = action[3 * leg_idx : 3 * (leg_idx + 1)]
            leg_q = self.compute_inverse_kinematics(leg_idx, xyz_desired)
            desired_joint_angles[3 * leg_idx : 3 * (leg_idx + 1)] = leg_q

        # Send desired joint positions.
        return dict(joint_pos=Float32MultiArray(data=desired_joint_angles))

    @staticmethod
    def compute_inverse_kinematics(leg_id: int, xyz_coord: np.ndarray):
        """Get joint angles for leg leg_id with desired xyz position in leg frame.

        Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;

        From SpotMicro:
        https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py
        """
        # rename links
        shoulder_length = go1_config.HIP_LINK_LENGTH
        elbow_length = go1_config.THIGH_LINK_LENGTH
        wrist_length = go1_config.CALF_LINK_LENGTH
        # coords
        x = xyz_coord[0]
        y = xyz_coord[1]
        z = xyz_coord[2]

        # get_domain
        D = (y**2 + (-z) ** 2 - shoulder_length**2 + (-x) ** 2 - elbow_length**2 - wrist_length**2) / (
            2 * wrist_length * elbow_length
        )

        D = np.clip(D, -1.0, 1.0)

        # check Right vs Left leg for hip angle
        sideSign = 1
        if leg_id == 0 or leg_id == 2:
            sideSign = -1

        # Right Leg Inverse Kinematics Solver
        wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
        sqrt_component = y**2 + (-z) ** 2 - shoulder_length**2
        if sqrt_component < 0.0:
            sqrt_component = 0.0
        shoulder_angle = -np.arctan2(z, y) - np.arctan2(np.sqrt(sqrt_component), sideSign * shoulder_length)
        elbow_angle = np.arctan2(-x, np.sqrt(sqrt_component)) - np.arctan2(
            wrist_length * np.sin(wrist_angle), elbow_length + wrist_length * np.cos(wrist_angle)
        )
        joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
        return joint_angles
