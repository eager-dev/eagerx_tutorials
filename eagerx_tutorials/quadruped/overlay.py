import eagerx
from eagerx import register, Space
from eagerx.utils.utils import Msg
from typing import List
import cv2
from matplotlib.cm import get_cmap
import numpy as np
from collections import deque
from functools import partial


class XyPlane(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: int = eagerx.ENVIRONMENT,
        color: str = "cyan",
        px_pm: int = 45,
        top_left: List[int] = None,
        lower_right: List[int] = None,
        num_eps: int = 20,
        colormap: str = "plasma",
    ) -> eagerx.specs.NodeSpec:
        """XyPlane spec"""
        spec = cls.get_specification()

        # Adjust default params
        spec.config.update(name=name, rate=rate, process=process, color=color, inputs=["position"], outputs=["image"])
        spec.config.px_pm = px_pm
        spec.config.colormap = colormap
        spec.config.top_left = top_left if isinstance(top_left, list) else [-3, -3]  # x,y in [m]
        spec.config.lower_right = lower_right if isinstance(lower_right, list) else [6, 8]  # x,y in [m]
        spec.config.num_eps = num_eps
        spec.inputs.position.window = 0  # Receive all new position messages since last callback

        # Calculate shape
        tl_x, tl_y = spec.config.top_left
        lr_x, lr_y = spec.config.lower_right
        width = int((lr_y - tl_y) * spec.config.px_pm)  # px
        height = int((lr_x - tl_x) * spec.config.px_pm)
        spec.outputs.image.space = Space(low=0, high=255, shape=(height, width, 3), dtype="uint8")
        return spec

    def initialize(self, spec):
        # Calculate width, height, and shape
        self.px_pm = spec.config.px_pm
        self.thickness = max(int(spec.config.px_pm * 0.10), 1)
        self.top_left = spec.config.top_left
        self.lower_right = spec.config.lower_right
        tl_x, tl_y = spec.config.top_left
        lr_x, lr_y = spec.config.lower_right
        self.width = int((lr_y - tl_y) * spec.config.px_pm)  # px
        self.height = int((lr_x - tl_x) * spec.config.px_pm)
        self.shape = (self.height, self.width)

        # Prepare history
        self.num_eps = spec.config.num_eps
        self.last_xy = deque(maxlen=spec.config.num_eps)
        self.colors = deque(maxlen=spec.config.num_eps)
        self.xy = None  # [m]

        # Prepare colormap
        self.cmap = get_cmap(spec.config.colormap)
        interp = np.linspace(0, 1, num=spec.config.num_eps + 1, dtype="float32")
        self.colormap = [self.cmap(i, bytes=True) for i in reversed(interp)]
        self.colormap = [(int(i[2]), int(i[1]), int(i[0])) for i in self.colormap]

        # Plot points
        self.plot_px = partial(
            self._plot_px,
            thickness=self.thickness,
            height=self.height,
            width=self.width,
            px_pm=spec.config.px_pm,
            tl_x=tl_x,
            tl_y=tl_y,
        )

    @register.states()
    def reset(self):
        # Add xy-coordinates from previous episode to buffer
        if self.xy is None:  # First reset
            self.xy = []
        if len(self.xy) > 1:
            if len(self.last_xy) == self.num_eps:
                self.colors = deque([i - 1 for i in self.colors], maxlen=self.num_eps)
                self.colors.append(len(self.last_xy) - 1)
            else:
                self.colors.append(len(self.last_xy))
            self.last_xy.append(self.xy)

        # Empty xy-coordinates
        self.xy = []  # [m]

        # Construct base image with xy-coordinates from previous episodes
        self.base_img = 255 * np.ones((self.height, self.width, 3), dtype="uint8")
        tl_x, tl_y = self.top_left
        lr_x, lr_y = self.lower_right
        self._plot_overlay(self.base_img, (0, 0, 0), 3, self.width, self.px_pm, tl_x, tl_y, lr_x, lr_y)
        for idx, xy_list in enumerate(self.last_xy):
            color = self.colormap[self.colors[idx]]
            # p = (self.num_eps - len(self.last_xy) + idx) / self.num_eps
            # opaque = tuple([int(255 - p * (255 - x)) for x in color])
            for xy in xy_list:
                self.plot_px(self.base_img, xy, color)

    @staticmethod
    def _plot_overlay(img, color, thickness, width, px_pm, tl_x, tl_y, lr_x, lr_y):
        # Put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Top-view (i.e. xy-plane)"
        text_size = cv2.getTextSize(text, font, 0.75, 2)[0]
        text_x = int((width - text_size[0]) / 2)
        text_y = int(text_size[1])
        img = cv2.putText(img, text, (text_x, text_y), font, 0.75, thickness=2, color=(0, 0, 0))

        center = (int(-tl_y * px_pm), int(-tl_x * px_pm))

        # x-axis
        x = center[0] + int(lr_y * px_pm * 0.75)
        cv2.arrowedLine(img, center, (x, center[1]), color, thickness, tipLength=0.1)
        text_x = (x + text_size[1], center[1] + text_size[1] // 2)
        cv2.putText(img, "x", text_x, font, 0.75, thickness=2, color=(0, 0, 0))
        # y-axis
        y = center[1] + int(lr_x * px_pm * 0.75)
        cv2.arrowedLine(img, center, (center[0], y), color, thickness, tipLength=0.1)
        text_y = (center[0] - text_size[1] // 2, y + text_size[1])
        cv2.putText(img, "y", text_y, font, 0.75, thickness=2, color=(0, 0, 0))

    @staticmethod
    def _plot_px(img, xy, color, thickness, height, width, px_pm, tl_x, tl_y):
        x, y = xy
        x -= tl_x
        y -= tl_y
        px_x = int(x * px_pm)
        px_y = int(y * px_pm)
        if px_y > height or px_x > width:
            return
        # Plot pixel
        cv2.circle(img, (px_y, px_x), thickness, color, -1)

    @register.inputs(position=Space(low=[-10.0, -10.0, 0.0], high=[10.0, 10.0, 2.0]))
    @register.outputs(image=Space(dtype="uint8"))
    def callback(self, t_n: float, position: Msg):
        # Select newest color
        idx = len(self.last_xy)
        color = self.colormap[idx]

        # plot points
        for xyz in position.msgs:
            xy = [xyz.data[0], xyz.data[1]]
            self.plot_px(self.base_img, xy, color)
            self.xy.append(xy)

        # Prepare image for transmission.
        return dict(image=self.base_img)
