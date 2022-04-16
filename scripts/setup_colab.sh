#!/bin/bash

echo 'Running on CoLab.'

if [ ! -f "/root/roscore" ]; then
  echo "#!/bin/bash \n" > /root/roscore
  echo "source /opt/ros/melodic/setup.bash" > /root/roscore
  echo "/opt/ros/melodic/bin/roscore" > /root/roscore
  chmod +x /root/roscore
  echo "/root/roscore created."
fi

if [ ! -f "/root/roslab" ]; then
  echo "#!/bin/bash \n" > /root/roslab
  echo "source /opt/ros/melodic/setup.bash" > /root/roslab
  echo "ROS_MASTER_URI="0.0.0.0"" > /root/roslab
  echo "exec $@" > /root/roslab
  chmod +x /root/roscore
  echo '/root/roslab created.'
fi

echo 'Execute ROS commands as "!~/roslab ...".'

ubuntu_version="$(lsb_release -r -s)"

if [ $ubuntu_version == "18.04" ]; then
  ROS_NAME="melodic"
#elif [ $ubuntu_version == "16.04" ]; then
#  ROS_NAME="kinetic"
#elif [ $ubuntu_version == "20.04" ]; then
#  ROS_NAME="noetic"
else
  echo -e "Unsupported Ubuntu version: $ubuntu_version"
  echo -e "This colab setup script only works with 18.04"
  exit 1
fi

if [ -d "/opt/ros/melodic" ]; then
  echo "Ros distribution already installed: ros-$ROS_NAME-desktop."
else
  start_time="$(date -u +%s)"

  echo "Ubuntu $ubuntu_version detected. ROS-$ROS_NAME chosen for installation.";

  echo -e "\e[1;33m ******************************************** \e[0m"
  echo -e "\e[1;33m The installation may take around 5  Minutes! \e[0m"
  echo -e "\e[1;33m ******************************************** \e[0m"
  sleep 4

  echo "deb http://packages.ros.org/ros/ubuntu bionic main" >> /etc/apt/sources.list.d/ros-latest.list
  echo "- deb added"

  apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 >> /tmp/key.txt
  echo "- key added"

  apt update >> /tmp/update.txt
  echo "- apt updated"

  apt install ros-melodic-desktop  > /tmp/ros_install.txt
  echo "- ROS-$ROS_NAME-desktop installed.";

  end_time="$(date -u +%s)"
  elapsed="$(($end_time-$start_time))"

  echo "ROS installation complete, took $elapsed seconds in total"
fi
