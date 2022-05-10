#!/bin/bash

if [ ! -f "/root/roscore" ]; then
  echo -e "#!/bin/bash \n" >> /root/roscore
  echo -e "source /opt/ros/melodic/setup.bash \n" >> /root/roscore
  echo -e "/opt/ros/melodic/bin/roscore" >> /root/roscore
  chmod +x /root/roscore
  echo "/root/roscore created."
fi

if [ ! -f "/root/roslab" ]; then
  echo -e "#!/bin/bash \n" >> /root/roslab
  echo -e 'ROS_MASTER_URI="0.0.0.0"\n' >> /root/roslab
  echo -e "source /opt/ros/melodic/setup.bash \n" >> /root/roslab
  echo -e "exec \$@" >> /root/roslab
  chmod +x /root/roslab
  echo '/root/roslab created.'
fi

echo 'Execute ROS commands as "!~/roslab ...".'

ubuntu_version="$(lsb_release -r -s)"

if [ $ubuntu_version == "18.04" ]; then
  ROS_NAME="melodic"
#elif [ $ubuntu_version == "16.04" ]; then&1
#  ROS_NAME="kinetic"
#elif [ $ubuntu_version == "20.04" ]; then
#  ROS_NAME="noetic"
else
  echo -e "Unsupported Ubuntu version: $ubuntu_version"
  echo -e "This colab setup script only works with 18.04"
  exit 1
fi

if [ -d "/opt/ros/melodic" ]; then
  echo "Ros distribution already installed: ros-$ROS_NAME-ros-base."
else
  start_time="$(date -u +%s)"

  echo "Ubuntu $ubuntu_version detected. ROS-$ROS_NAME chosen for installation.";

  echo -e "\e[1;33m ******************************************** \e[0m"
  echo -e "\e[1;33m The installation may take around 3  Minutes! \e[0m"
  echo -e "\e[1;33m ******************************************** \e[0m"
  sleep 4

  {
    echo "deb http://packages.ros.org/ros/ubuntu bionic main" >> /etc/apt/sources.list.d/ros-latest.list 2>/tmp/ros_install.txt
  } || {
    echo "ROS installation failed. Check the log in /tmp/ros_install.txt."
    exit 1
  }
  echo "- deb added."

  {
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 >> /tmp/ros_install.txt 2>&1
  } || {
    echo "ROS installation failed. Check the log in /tmp/ros_install.txt."
    exit 1
  }
  echo "- key added."

  {
    apt-key del 7fa2af80 >> /tmp/ros_install.txt 2>&1
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub >> /tmp/ros_install.txt 2>&1
    apt update >> /tmp/ros_install.txt 2>&1
  } || {
    echo "ROS installation failed. Check the log in /tmp/ros_install.txt."
    exit 1
  }
  echo "- apt updated."

  {
    apt install ros-melodic-ros-base  > /tmp/ros_install.txt 2>&1
    apt-get install ros-melodic-cv-bridge > /tmp/ros_install.txt 2>&1
  } || { 
    echo "ROS installation failed. Check the log in /tmp/ros_install.txt."
    exit 1
  }
  echo "- ROS-$ROS_NAME-ros-base installed.";

  end_time="$(date -u +%s)"
  elapsed="$(($end_time-$start_time))"

  echo "ROS installation complete, took $elapsed seconds in total."
  
  echo 'Uninstalling paramiko with pip.' && pip2 uninstall -qy paramiko && pip2 install -q paramiko==2.10.4
fi

