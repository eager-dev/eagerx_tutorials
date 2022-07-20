"""
CPG in polar coordinates based on:
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306
Original author: Guillaume Bellegarda
"""

from typing import Optional
import numpy as np
import eagerx
from eagerx import Node, Space
from eagerx import register
from eagerx.utils.utils import Msg

from eagerx_tutorials.quadruped.hopf_network import HopfNetwork
import eagerx_tutorials.quadruped.go1.configs_go1 as go1_config


class CpgGait(Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        gait: str,
        omega_swing: float,
        omega_stance: float,
        ground_clearance: float = 0.04,
        ground_penetration: float = 0.02,
        mu: int = 2,
        couple: bool = True,
        coupling_strength: float = 1.0,
        robot_height: float = 0.25,
        des_step_len: float = 0.04,
        process: Optional[int] = eagerx.ENVIRONMENT,
    ) -> eagerx.specs.NodeSpec:
        """Make a spec to create a CpgGait node that produces a quadruped gait.

        It uses a CPG network based on hopf polar equations mapped to foot positions in Cartesian space.

        :param name: User specified node name.
        :param rate: Rate (Hz) at which the callback is called.
        :param gait: Change depending on desired gait.
        :param omega_swing: todo: MUST EDIT
        :param omega_stance:  todo: MUST EDIT
        :param ground_clearance: Foot swing height.
        :param ground_penetration: Foot stance penetration into ground.
        :param mu: todo: 1**2, converge to sqrt(mu)
        :param couple: Should couple.
        :param coupling_strength: Coefficient to multiply coupling matrix.
        :param robot_height: In nominal case (standing).
        :param des_step_len: Desired step length.
        :param process: Process in which this node is launched. See :class:`~eagerx.core.constants.process` for all options.
        :return: NodeSpec
        """
        spec = cls.get_specification()

        # Modify default params
        spec.config.update(name=name, rate=rate, process=process, inputs=["offset"], outputs=["cartesian_pos", "xs_zs"])

        # Modify params (args to .initialize())
        spec.config.update(mu=mu, gait=gait, omega_swing=omega_swing, omega_stance=omega_stance)
        spec.config.update(ground_clearance=ground_clearance, ground_penetration=ground_penetration)
        spec.config.update(couple=couple, coupling_strength=coupling_strength)
        spec.config.update(robot_height=robot_height, des_step_len=des_step_len)
        return spec

    def initialize(self, spec):
        assert spec.config.gait == "TROT", "xs_zs is only correct for TROT gait."
        self.n_legs = 4
        self.side_sign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)
        self.foot_y = 0.0838  # this is the hip length
        self.cpg = HopfNetwork(
            mu=spec.config.mu,
            gait=spec.config.gait,
            omega_swing=spec.config.omega_swing,
            omega_stance=spec.config.omega_stance,
            time_step=0.005,  # Always update cpg with 200 Hz.
            ground_clearance=spec.config.ground_clearance,  # foot swing height
            ground_penetration=spec.config.ground_penetration,  # foot stance penetration into ground
            robot_height=spec.config.robot_height,  # in nominal case (standing)
            des_step_len=spec.config.des_step_len,  # 0 for jumping
        )

    @register.states()
    def reset(self):
        self.cpg.reset()

    @register.inputs(offset=Space(low=[-0.01] * 4, high=[0.01] * 4))
    @register.outputs(
        cartesian_pos=Space(shape=(len(go1_config.NOMINAL_FOOT_POS_LEG_FRAME),), dtype="float32"),  # TODO: Set correct bounds
        xs_zs=Space(
            low=[-0.05656145, -0.26999995, -0.05656852, -0.2699973], high=[0.05636625, -0.21000053, 0.05642071, -0.21001561]
        ),
    )
    def callback(self, t_n: float, offset: Msg):
        # update CPG
        while self.cpg.t <= t_n:
            self.cpg.update()

        # get desired foot positions from CPG
        xs, zs = self.cpg.get_xs_zs()

        # get unique xs & zs positions (BASED ON TROT)
        unique_xs_zs = np.array([xs[0], zs[0], xs[1], zs[1]], dtype="float32")

        action = np.zeros((12,), dtype="float32")
        offset = offset.msgs[-1].data
        for i in range(self.n_legs):
            xyz_desired = np.array([xs[i], self.side_sign[i] * self.foot_y + offset[i], zs[i]])
            action[3 * i : 3 * i + 3] = xyz_desired
        return dict(cartesian_pos=action, xs_zs=unique_xs_zs)
