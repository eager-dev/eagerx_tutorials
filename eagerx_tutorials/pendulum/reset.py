from typing import Optional, List
import numpy as np
import eagerx
from eagerx import Space
from eagerx.utils.utils import Msg


def wrap_angle(angle):
    return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))


class ResetAngle(eagerx.ResetNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        threshold: float = 0.1,
        timeout: float = 5.0,
        gains: Optional[List[float]] = None,
        u_range: Optional[List[float]] = None,
    ):
        """This AngleReset node resets the pendulum to a desired angle with zero angular velocity. NOte that this controller
        only works properly when resetting the pendulum near the downward facing equilibrium.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param threshold: Absolute difference between the desired and goal state, before considering the reset complete.
        :param timeout: Maximum time (seconds) before considering the reset finished (regardless whether the goal was reached).
        :param gains: Gains of the PID controller used to reset.
        :param u_range: Min and max action.
        :return:
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.ENVIRONMENT, color="grey")
        spec.config.update(inputs=["theta", "dtheta"], targets=["goal"], outputs=["u"])
        spec.config.update(u_range=u_range, threshold=threshold, timeout=timeout)
        spec.config.gains = gains if isinstance(gains, list) else [1.0, 0.5, 0.0]

        # Add space_converter
        spec.outputs.u.space = Space(low=[u_range[0]], high=[u_range[1]], dtype="float32")
        return spec

    def initialize(self, spec):
        self.threshold = spec.config.threshold
        self.timeout = spec.config.timeout
        self.u_min, self.u_max = spec.config.u_range

        # Creat a simple PID controller
        from eagerx_tutorials.pendulum.pid import PID

        gains = spec.config.gains
        self.controller = PID(u0=0.0, kp=gains[0], kd=gains[1], ki=gains[2], dt=1 / self.rate)

    @eagerx.register.states()
    def reset(self):
        # Reset the internal state of the PID controller (ie the error term).
        self.controller.reset()
        self.ts_start_routine = None

    @eagerx.register.inputs(
        theta=Space(shape=(), dtype="float32"),
        dtheta=Space(shape=(), dtype="float32"),
    )
    @eagerx.register.targets(goal=Space(low=[-3.14, -9.0], high=[3.14, 9.0]))
    @eagerx.register.outputs(u=Space(low=[-2.0], high=[2.0]))
    def callback(self, t_n: float, goal: Msg, theta: Msg = None, dtheta: Msg = None, x: Msg = None):
        if self.ts_start_routine is None:
            self.ts_start_routine = t_n

        # Convert messages to floats and numpy array
        theta = theta.msgs[-1]  # Take the last received message
        dtheta = dtheta.msgs[-1]  # Take the last received message
        goal = np.array(goal.msgs[-1], dtype="float32")  # Take the last received message

        # Define downward angle as theta=0 (resolve downward discontinuity)
        theta += np.pi
        goal[0] += np.pi

        # Wrap angle between [-pi, pi]
        theta = wrap_angle(theta)
        goal[0] = wrap_angle(goal[0])

        # Overwrite the desired velocity to be zero.
        goal[1] = 0.0

        # Calculate the action using the PID controller
        u = self.controller.next_action(theta, ref=goal[0])
        u = np.clip(u, self.u_min, self.u_max)  # Clip u to range

        # Determine if we have reached our goal state
        done = np.isclose(np.array([theta, dtheta]), goal, atol=self.threshold).all()

        # If the reset routine takes too long, we timeout the routine and simply assume that we are done.
        done = done.item() or (t_n - self.ts_start_routine) > self.timeout

        # Prepare output message for transmission.
        # This must contain a message for every registered & selected output and target.
        # For targets, this message decides whether the goal state has been reached (or we, for example, timeout the reset).
        # The name for this target message is the registered target name + "/done".
        output_msgs = {"u": np.array([u], dtype="float32"), "goal/done": done}
        return output_msgs
