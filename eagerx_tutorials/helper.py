import os
import site
import importlib


def setup_notebook():
    if "google.colab" in str(get_ipython()):  # noqa:
        print("Running on CoLab")
        # Required to source ros in new processes
        os.environ["EAGERX_COLAB"] = "1"
        # Set paths to ROS libraries (instead of sourcing)
        site.addsitedir("/opt/ros/melodic/lib/python2.7/dist-packages")
        site.addsitedir("/usr/lib/python2.7/dist-packages")
        os.environ["ROS_PACKAGE_PATH"] = "/opt/ros/melodic/share/"

        # No restart possible.
        from datetime import datetime
        import subprocess

        try:
            roscore.terminate()
            print(f"[{datetime.today()}] Roscore restarted!")
        except NameError:
            print(f"[{datetime.today()}] Roscore started!")

        # Start the roscore node
        cmd = get_tutorial_path() + "/../scripts/roscore"
        roscore = subprocess.Popen([cmd])  # noqa:
    else:
        print("Not running on CoLab")

    os.environ["EAGERX_RELOAD"] = "1"


def get_tutorial_path():
    spec = importlib.util.find_spec("eagerx_tutorials")
    return os.path.dirname(spec.origin)
