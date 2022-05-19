import os
import site
from datetime import datetime
import subprocess
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from pathlib import Path
import base64
from IPython import display as ipythondisplay


def setup_notebook():
    if "google.colab" in str(get_ipython()):  # noqa:
        print("Running on CoLab.")
        # Required to source ros in new processes
        os.environ["EAGERX_COLAB"] = "1"
        # Set paths to ROS libraries (instead of sourcing)
        site.addsitedir("/opt/ros/melodic/lib/python2.7/dist-packages")
        site.addsitedir("/usr/lib/python2.7/dist-packages")
        os.environ["ROS_PACKAGE_PATH"] = "/opt/ros/melodic/share/"
    else:
        print("Not running on CoLab.")
        print('Execute ROS commands as "!...".')

    os.environ["EAGERX_RELOAD"] = "1"

    try:
        import rospy, roslaunch, rosparam, rospkg  # noqa:

        pkg = rospkg.RosStack()
        if "noetic" in pkg.get_path("ros"):
            ros_version = "noetic"
        elif "melodic" in pkg.get_path("ros"):
            ros_version = "melodic"
        else:
            raise ModuleNotFoundError("No ROS version (noetic, melodic) available. Check if installed & sourced.")
        print(f"ROS {ros_version} available.")
    except ModuleNotFoundError:
        ModuleNotFoundError("No ROS version (noetic, melodic) available. Check if installed & sourced.")

    # No restart possible.
    if "google.colab" in str(get_ipython()):  # noqa:
        timestamp = datetime.today().strftime("%H:%M:%S")
        try:
            roscore.terminate()
            print(f"Roscore restarted: {timestamp}.")
        except NameError:
            print(f"Roscore started: {timestamp}.")

        # Start the roscore node
        roscore = subprocess.Popen(["bash", "/root/roscore"])  # noqa:


def record_video(env, model, video_length=500, prefix="", video_folder="videos/"):
    """
    Adapted from https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb

    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    # Set up fake display; otherwise rendering will fail
    os.system("Xvfb :1 -screen 0 1024x768x24 &")
    os.environ["DISPLAY"] = ":1"

    eval_env = DummyVecEnv([lambda: env])

    # Start the video at step=0 and record for video length
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


def show_video(video_folder="videos/", prefix=""):
    """
    Adapted from https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb

    which was taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_folder).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
