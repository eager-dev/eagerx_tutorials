import os
import site
from datetime import datetime
import subprocess


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
