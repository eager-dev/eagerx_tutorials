import eagerx
from eagerx import register
from eagerx.utils.utils import Msg
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
import cv2
import numpy as np


class Overlay(eagerx.Node):
    @staticmethod
    @register.spec("Overlay", eagerx.Node)
    def spec(
        spec,
        name: str,
        rate: float,
        process: int = eagerx.process.ENVIRONMENT,
        color: str = "cyan",
    ):
        """Overlay spec"""
        # Fills spec with defaults parameters
        spec.initialize(Overlay)

        # Adjust default params
        spec.config.update(
            name=name, rate=rate, process=process, color=color, inputs=["base_image", "u", "theta"], outputs=["image"]
        )

    def initialize(self):
        pass

    @register.states()
    def reset(self):
        pass

    def _convert_to_cv_image(self, img):
        if isinstance(img.data, bytes):
            cv_image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        else:
            cv_image = np.array(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        if "rgb" in img.encoding:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image

    @register.inputs(base_image=Image, u=Float32MultiArray, theta=Float32)
    @register.outputs(image=Image)
    def callback(self, t_n: float, base_image: Msg, u: Msg, theta: Msg):
        if len(base_image.msgs[-1].data) > 0:
            u = u.msgs[-1].data[0] if u else 0
            theta = theta.msgs[-1].data

            # Set background image from base_image
            img = self._convert_to_cv_image(base_image.msgs[-1])
            width = base_image.msgs[-1].width
            height = base_image.msgs[-1].height
            side_length = min(width, height)

            # Put text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Applied Voltage"
            text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
            text_x = int((width - text_size[0]) / 2)
            text_y = int(text_size[1])
            # print(f"text_size = {text_size} | text_x = {text_x} | text_y = {text_y}")
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

            # Prepare image for transmission.
            data = img.tobytes("C")
            msg = Image(data=data, height=height, width=width, encoding="bgr8", step=3 * width)
            return dict(image=msg)
        else:
            return dict(image=Image())
