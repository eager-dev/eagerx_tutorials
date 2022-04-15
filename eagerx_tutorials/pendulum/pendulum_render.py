import cv2
import numpy as np
from datetime import datetime


def pendulum_render_fn(img, observation, action):
    height, width, _ = img.shape
    state = observation.msgs[-1].data

    img += 255
    l = width // 3
    sin_theta, cos_theta = np.sin(state[0]), np.cos(state[0])

    # Draw pendulum
    img = cv2.line(
        img,
        (width // 2, height // 2),
        (width // 2 + int(l * sin_theta), height // 2 - int(l * cos_theta)),
        (0, 0, 255),
        15,
    )

    # Draw mass
    img = cv2.circle(img, (width // 2 + int(l * sin_theta), height // 2 - int(l * cos_theta)), 30, (0, 0, 0), -1)

    # Draw velocity vector
    img = cv2.arrowedLine(
        img,
        (width // 2 + int(l * sin_theta), height // 2 - int(l * cos_theta)),
        (
            width // 2 + int(l * (sin_theta + state[1] * cos_theta / 5)),
            height // 2 + int(l * (-cos_theta + state[1] * sin_theta / 5)),
        ),
        (0, 0, 0),
        2,
    )

    # Draw voltage
    if action.msgs is not None and action.msgs[-1].data is not None:
        action = action.msgs[-1].data[0]
    else:
        action = 0

    # Put text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Applied Voltage"
    text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
    text_x = int((width - text_size[0]) / 2)
    text_y = int(text_size[1])
    img = cv2.putText(img, text, (text_x, text_y), font, 0.5, (0, 0, 0))

    # Draw grey bar
    img = cv2.rectangle(img, (width // 10, height // 20), (9 * width // 10, 2 * height // 20), (125, 125, 125), -1)

    # Fill black black bar proportional to the action that is applied
    p1 = (width // 2, height // 20)
    p2 = (int(width / 2 * (1 + action * 4 / 15)), 2 * height // 20)
    img = cv2.rectangle(img, p1, p2, (255, 0, 0), -1)

    # Draw line at 0
    img = cv2.line(img, p1, (width // 2, 2 * height // 20), (0, 0, 0))

    # Draw joint
    img = cv2.circle(img, (width // 2, height // 2), 3, (0, 0, 0), -1)

    # save image
    time = datetime.now()
    cv2.imwrite("figures/pendulum_" + str(time) + ".jpg", img)
    return img
