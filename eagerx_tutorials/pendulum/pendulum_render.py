import cv2
import numpy as np


def pendulum_render_fn(img, observation, action):
    height, width, _ = img.shape
    side_length = min(width, height)
    state = observation.msgs[-1].data

    img += 255
    length = side_length // 3
    sin_theta, cos_theta = np.sin(state[0]), np.cos(state[0])

    # Draw pendulum
    img = cv2.line(
        img,
        (width // 2, height // 2),
        (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)),
        (0, 0, 255),
        max(side_length // 32, 1),
    )

    # Draw mass
    img = cv2.circle(
        img, (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)), side_length // 16, (0, 0, 0), -1
    )

    # Draw velocity vector
    img = cv2.arrowedLine(
        img,
        (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)),
        (
            width // 2 + int(length * (sin_theta + state[1] * cos_theta / 5)),
            height // 2 + int(length * (-cos_theta + state[1] * sin_theta / 5)),
        ),
        (0, 0, 255),
        max(side_length // 240, 1),
    )

    # Draw voltage
    if len(action.msgs) > 0 and len(action.msgs[-1].data) > 0:
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
    img = cv2.rectangle(
        img,
        (width // 2 - side_length * 4 // 10, height // 2 - side_length * 9 // 20),
        (width // 2 + 4 * side_length // 10, height // 2 - 4 * side_length // 10),
        (125, 125, 125),
        -1,
    )

    # Fill bar proportional to the action that is applied
    p1 = (width // 2, height // 2 - side_length * 9 // 20)
    p2 = (width // 2 + int(side_length * action * 2 / 15), height // 2 - 4 * side_length // 10)
    img = cv2.rectangle(img, p1, p2, (0, 0, 0), -1)

    # Draw joint
    img = cv2.circle(img, (width // 2, height // 2), max(side_length // 160, 1), (0, 0, 0), -1)

    return img
