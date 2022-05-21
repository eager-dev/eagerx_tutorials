

if __name__ == "__main__":
    # Initialize eagerx (starts roscore if not already started.)
    import eagerx
    eagerx.initialize("eagerx_core", log_level=eagerx.log.INFO)

    # Parameters
    use_image = False
    rate = 20
    render_rate = 5
    episode_timeout = 10  # [s]
    train_episodes = 30

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create robot
    import eagerx_tutorials.quadruped.object  # noqa: F401
    sensors = ["joint_position", "joint_velocity", "orientation", "position", "velocity", "force_torque"]
    robot = eagerx.Object.make("Quadruped", "quadruped", actuators=["joint_control"], sensors=sensors, rate=rate)
    robot.actuators.joint_control.rate = 200
    graph.add(robot)

    # Create cartesian control node
    import eagerx_tutorials.quadruped.cartesian_control  # noqa: F401
    cartesian_control = eagerx.Node.make("CartesiandPDController", "cartesian_control", rate=200)
    graph.add(cartesian_control)

    # Create cpg node
    import eagerx_tutorials.quadruped.cpg_gait  # noqa: F401
    cpg = eagerx.Node.make("CpgGait", "cpg", rate=200, gait="TROT", omega_swing=16 * 3.14, omega_stance=4 * 3.14)
    graph.add(cpg)

    # Connect graph
    use_sensors = ["orientation", "force_torque", "xs_zs"]
    graph.connect(action="offset", target=cpg.inputs.offset)
    graph.connect(source=cpg.outputs.cartesian_pos, target=cartesian_control.inputs.cartesian_pos)
    graph.connect(source=cartesian_control.outputs.joint_pos, target=robot.actuators.joint_control)
    # if "joint_position" in use_sensors:
    graph.connect(observation="joint_position", source=robot.sensors.joint_position)
    # if "joint_velocity" in use_sensors:
    graph.connect(observation="joint_velocity", source=robot.sensors.joint_velocity)
    # if "position" in use_sensors:
    graph.connect(observation="position", source=robot.sensors.position)
    # if "force_torque" in use_sensors:
    graph.connect(observation="force_torque", source=robot.sensors.force_torque)
    # if "velocity" in use_sensors:
    graph.connect(observation="velocity", source=robot.sensors.velocity)
    # if "xs_zs" in use_sensors:
    initial_obs = [-0.01354526, -0.26941818, 0.0552178, -0.25434446]
    graph.connect(observation="xs_zs", source=cpg.outputs.xs_zs, skip=True, initial_obs=initial_obs)
    assert 'orientation' in use_sensors, "We require the orientation sensor to calculate the yaw rate."
    graph.connect(observation="orientation", source=robot.sensors.orientation, window=2)  # window=2

    # Select rendering
    if use_image:
        graph.add_component(robot.sensors.image)
        graph.render(robot.sensors.image, rate=render_rate)
    else:
        # Create xy-plane visualization node
        import eagerx_tutorials.quadruped.overlay # noqa Registers the overlay node
        xy_plane = eagerx.Node.make("XyPlane", "xy_plane", rate=render_rate)
        graph.add(xy_plane)
        graph.connect(source=robot.sensors.position, target=xy_plane.inputs.position)
        graph.render(xy_plane.outputs.image, rate=render_rate)

    # Define engine
    engine = eagerx.Engine.make("PybulletEngine", rate=200, gui=True, egl=True, process=eagerx.process.ENVIRONMENT)

    # Define Gym Environment
    import pybullet
    import numpy as np
    import gym
    from typing import Dict, Tuple

    class QuadrupedEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, episode_timeout):
            super(QuadrupedEnv, self).__init__(name, rate, graph, engine, force_start=True)
            self.steps = None
            self.timeout_steps = int(episode_timeout * rate)
            self.rate = rate
            self.desired_yaw_rate = np.deg2rad(20)

        @property
        def observation_space(self) -> gym.spaces.Dict:
            return self._observation_space

        @property
        def action_space(self) -> gym.spaces.Dict:
            return self._action_space

        def reset(self):
            # Reset number of steps
            self.steps = 0

            # Sample desired states
            states = self.state_space.sample()

            # set view location of camera
            states["quadruped/image/pos"] = np.array([-1, -1, 0.5])
            orientation = list(pybullet.getQuaternionFromEuler(np.deg2rad([-90, 0, 0])))
            states["quadruped/image/orientation"] = np.array(orientation)

            # Perform reset
            obs = self._reset(states)
            return obs

        def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
            obs = self._step(action)
            self.steps += 1

            # Convert Quaternion to Euler
            _, _, prev_yaw = pybullet.getEulerFromQuaternion(obs["orientation"][-2])
            roll, pitch, yaw = pybullet.getEulerFromQuaternion(obs["orientation"][-1])

            # Current angular velocity
            yaw_rate = (yaw - prev_yaw) * self.rate

            # Calculate reward
            yaw_cost = (yaw_rate - self.desired_yaw_rate) ** 2
            alive_bonus = 0.25
            reward = alive_bonus - yaw_cost

            # Determine termination condition
            has_fallen = abs(np.rad2deg(roll)) > 40 or abs(np.rad2deg(pitch)) > 40
            timeout = self.steps >= self.timeout_steps

            # Determine done flag
            done = timeout or has_fallen
            # Set info about episode truncation
            info = {"TimeLimit.truncated": timeout and not has_fallen}
            return obs, reward, done, info

    # Initialize Environment
    env = QuadrupedEnv(name="QuadEnv", rate=rate, graph=graph, engine=engine, episode_timeout=episode_timeout)

    # Stable-baselines
    from sb3_contrib import TQC
    from eagerx.wrappers import Flatten
    hyperparams = dict(
        learning_rate=1e-3,
        tau=0.02,
        gamma=0.98,
        buffer_size=300000,
        learning_starts=100,
        use_sde=True,
        use_sde_at_warmup=True,
        train_freq=8,
        gradient_steps=10,
        verbose=1,
        top_quantiles_to_drop_per_net=0,
        policy_kwargs=dict(n_critics=1, net_arch=dict(pi=[64, 64], qf=[64, 64])),
    )
    model = TQC("MlpPolicy", Flatten(env), **hyperparams)

    # Train
    try:
        train_steps = int(train_episodes * episode_timeout * rate)
        env.render("human")
        model.learn(train_steps)
        model.save("last_policy")
    except KeyboardInterrupt:
        model.save("last_policy")

    # Evaluate
    from eagerx_tutorials.quadruped.evaluate import EvaluateEnv
    from stable_baselines3.common.evaluation import evaluate_policy

    engine = eagerx.Engine.make("PybulletEngine", rate=200, gui=False, egl=False, process=eagerx.process.ENVIRONMENT)
    eval_env = EvaluateEnv(env, engine, episode_timeout=40, render="pybullet")
    eval_env.render("human")
    path = "last_policy"
    # path = "logs/Quadruped_1/rl_model_6000_steps"
    model = TQC.load(path)
    mean_reward, std = evaluate_policy(model, Flatten(eval_env), n_eval_episodes=5)
    print(f"Mean reward = {mean_reward:.2f} +/- {std}")


