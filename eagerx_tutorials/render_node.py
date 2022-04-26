# import dependencies
from eagerx_tutorials.colab_render import InlineRender
import eagerx
import numpy as np
import rospy
import time
import cv2
from std_msgs.msg import UInt64, Bool
from sensor_msgs.msg import Image
from typing import Optional, List


class ColabRenderNode(eagerx.Node):
    @staticmethod
    @eagerx.register.spec("ColabRenderNode", eagerx.Node)
    def spec(
        spec: eagerx.specs.NodeSpec,
        rate: int,
        process: int = eagerx.process.ENVIRONMENT,
        fps: int = 25,
        shape: Optional[List[int]] = None,
        maxlen: int = 200,
        subsample: bool = True,
        log_level: int = eagerx.log.WARN,
        color: str = "grey",
    ):
        """ColabRenderNode spec"""
        # Initialize spec
        spec.initialize(ColabRenderNode)

        # Modify default node params
        spec.config.name = "env/render"
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.log_level = log_level
        spec.config.inputs = ["image"]
        spec.config.outputs = ["done"]
        spec.config.states = []

        # Custom params
        spec.config.fps = fps
        spec.config.shape = shape if isinstance(shape, list) else [64, 64]
        spec.config.maxlen = maxlen
        spec.config.subsample = True

        # Pre-set window
        spec.inputs.image.window = 0

    def initialize(self, fps, size, maxlen, shape, subsample):
        # todo: Overwrite fps if higher than rate
        # todo: Subsample if fps lower than rate * real_time_factor
        # todo: set node_fps either slightly higher or lower than js_fps?
        # todo: Avoid overflowing buffer
        self.dt_fps = 1 / fps
        self.subsample = subsample
        self.window = InlineRender(fps=fps, maxlen=maxlen, shape=shape)
        self.last_image = Image(data=[])
        self.render_toggle = False
        self.sub_toggle = rospy.Subscriber("%s/%s/toggle" % (self.ns, self.name), Bool, self._set_render_toggle)
        self.sub_get = rospy.Subscriber("%s/%s/get_last_image" % (self.ns, self.name), Bool, self._get_last_image)
        self.pub_set = rospy.Publisher(
            "%s/%s/set_last_image" % (self.ns, self.name),
            Image,
            queue_size=0,
            latch=True,
        )

    def _set_render_toggle(self, msg):
        if msg.data:
            rospy.loginfo("START RENDERING!")
        else:
            rospy.loginfo("STOP RENDERING!")
        self.render_toggle = msg.data

    def _get_last_image(self, msg):
        self.pub_set.publish(self.last_image)

    def reset(self):
        self.last_image = Image(data=[])

    @eagerx.register.inputs(image=Image)
    @eagerx.register.outputs(done=UInt64)
    def callback(self, t_n: float, image: Optional[eagerx.utils.utils.Msg] = None):
        # Fill output_msg with 'done' output --> signals that we are done rendering
        output_msgs = dict(done=UInt64())
        # Grab latest image
        if len(image.msgs) > 0:
            self.last_image = image.msgs[-1]
        # If too little time has passed, do not add frame (avoid buffer overflowing)
        if not time.time() > (self.dt_fps + self.window.timestamp):
            return output_msgs
        # Check if frame is not empty
        empty = self.last_image.height == 0 or self.last_image.width == 0
        if not empty and self.render_toggle:
            # Convert image to np array
            if isinstance(self.last_image.data, bytes):
                img = np.frombuffer(self.last_image.data, dtype=np.uint8).reshape(
                    self.last_image.height, self.last_image.width, -1
                )
            else:
                img = np.array(self.last_image.data, dtype=np.uint8).reshape(self.last_image.height, self.last_image.width, -1)
            # Convert to rgb (from bgr)
            if "rgb" in self.last_image.encoding:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Add image to buffer (where it is send async to javascript window)
            self.window.buffer_images(img)
        return output_msgs

    def shutdown(self):
        rospy.logdebug(f"[{self.name}] {self.name}.shutdown() called.")
        self.sub_toggle.unregister()
        self.sub_get.unregister()
        self.pub_set.unregister()
