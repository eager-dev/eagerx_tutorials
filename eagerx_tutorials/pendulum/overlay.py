import eagerx
from eagerx import register, Space
from eagerx.utils.utils import Msg
import cv2
import numpy as np


class Overlay(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: int = eagerx.process.ENVIRONMENT,
        color: str = "cyan",
    ):
        """Overlay spec"""
        spec = cls.get_specification()

        # Adjust default params
        spec.config.update(
            name=name, rate=rate, process=process, color=color, inputs=["base_image", "u", "theta"], outputs=["image"]
        )
        return spec

    def initialize(self, spec):
        pass

    @register.states()
    def reset(self):
        pass

    @register.inputs(
        base_image=Space(dtype="uint8"),
        u=Space(low=[-3], high=[3]),
        theta=Space(shape=(), dtype="float32"),
    )
    @register.outputs(image=Space(dtype="uint8"))
    def callback(self, t_n: float, base_image: Msg, u: Msg, theta: Msg):
        if len(base_image.msgs[-1].data) > 0:
            u = u.msgs[-1].data[0] if u else 0
            theta = theta.msgs[-1]

            # Set background image from base_image
            img = base_image.msgs[-1]
            height, width, _ = img.shape
            side_length = min(width, height)

            # Put text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Applied Voltage"
            text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
            text_x = int((width - text_size[0]) / 2)
            text_y = int(text_size[1])
            img = cv2.putText(img, text, (text_x, text_y), font, 0.5, (0, 0, 0))

            # Draw grey bar
            img = cv2.rectangle(
                img,
                (width // 2 - side_length * 4 // 10, height // 2 - side_length * 9 // 20),
                (width // 2 + 4 * side_length // 10, height // 2 - 4 * side_length // 10),
                (125, 125, 125),
                -1,
            )

            # Fill bar proportional to the action that is applied
            p1 = (width // 2, height // 2 - side_length * 9 // 20)
            p2 = (width // 2 + int(side_length * u * 2 / 15), height // 2 - 4 * side_length // 10)
            img = cv2.rectangle(img, p1, p2, (0, 0, 0), -1)

            # START EXERCISE 1.3
            img = cv2.putText(img, f"t ={t_n: .2f} s", (text_x, height - text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            # START EXERCISE 1.3

            # Add theta info
            img = cv2.putText(
                img, f"theta ={theta: .2f} rad", (text_x, height - int(2.2 * text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
            )

            return dict(image=img)
        else:
            return dict(image=np.zeros((0, 0, 3), dtype="uint8"))
