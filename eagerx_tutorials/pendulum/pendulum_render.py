import cv2
import numpy as np


def pendulum_render_fn(img, observation, action):
    height, width, _ = img.shape
    state = observation.msgs[-1].data

    img += 255
    l = width // 3
    sin_theta, cos_theta = np.sin(state[0]), np.cos(state[0])
    img = cv2.line(
        img,
        (width // 2, height // 2),
        (width // 2 + int(l * sin_theta), height // 2 - int(l * cos_theta)),
        (0, 0, 0),
        15,
    )
    img = cv2.arrowedLine(
        img,
        (width // 2 + int(l * sin_theta), height // 2 - int(l * cos_theta)),
        (width // 2 + int(l * (sin_theta + state[1] * cos_theta / 2)),
         height // 2 + int(l * (-cos_theta + state[1] * sin_theta / 2))),
        (0, 0, 255),
        2,
    )
    return img