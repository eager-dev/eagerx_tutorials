import os
import site


def setup_notebook():
    if 'google.colab' in str(get_ipython()):
        print('Running on CoLab')
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
            master.terminate()
            print(f"[{datetime.today()}] Roscore restarted!")
        except NameError as e:
            print(f"[{datetime.today()}] Roscore started!")

        # Start the master node
        master = subprocess.Popen(["/content/roscore"])
    else:
        print('Not running on CoLab')

    os.environ["EAGERX_RELOAD"] = "1"