import cv2
import numpy as np
import pyautogui


def find_game_area():
    """
    Detects and returns the cropped game area image and its bounding box.
    Returns:
        game_area_img: Cropped image of the game area (or None if not found)
        bbox: (x, y, w, h) bounding box of the game area (or None if not found)
    """
    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    _, game_area_mask = cv2.threshold(l_channel, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    game_area_mask = cv2.morphologyEx(game_area_mask, cv2.MORPH_CLOSE, kernel)
    game_area_mask = cv2.morphologyEx(game_area_mask, cv2.MORPH_OPEN, kernel)
    contours_game, _ = cv2.findContours(
        game_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours_game:
        largest_game_area = max(contours_game, key=cv2.contourArea)
        area = cv2.contourArea(largest_game_area)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(largest_game_area)
            print(f"Game area detected at: {(x, y, w, h)}")
            game_area_img = img[y : y + h, x : x + w]
            return game_area_img, (x, y, w, h)
    print("Could not detect game area. Make sure it's visible and green.")
    return None, None


def detect_game_objects(image):
    """
    Detects the snake head, snake body, and apple in the given image.
    Returns:
        results: dict with keys 'game_area', 'snake_head', 'snake_body', 'apple'
        output: annotated image
        mask_snake: binary mask of the snake (blue)
    """
    output = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    results = {"game_area": None, "snake_head": [], "snake_body": [], "apple": []}

    # Detect Snake (Blue)
    lower_blue = np.array([100, 120, 50])
    upper_blue = np.array([130, 255, 255])
    mask_snake = cv2.inRange(hsv, lower_blue, upper_blue)
    contours_snake, _ = cv2.findContours(
        mask_snake, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours_snake:
        largest_snake_contour = max(contours_snake, key=cv2.contourArea)
        snake_area = cv2.contourArea(largest_snake_contour)
        if snake_area > 100:
            snake_x, snake_y, snake_w, snake_h = cv2.boundingRect(largest_snake_contour)
            snake_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(snake_mask, [largest_snake_contour], -1, 255, -1)
            gray_roi = cv2.cvtColor(
                image[snake_y : snake_y + snake_h, snake_x : snake_x + snake_w],
                cv2.COLOR_BGR2GRAY,
            )
            _, eye_thresh = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
            eye_contours, _ = cv2.findContours(
                eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if eye_contours:
                leftmost_eye_x = snake_w
                for eye_cnt in eye_contours:
                    if cv2.contourArea(eye_cnt) > 10:
                        eye_x, _, _, _ = cv2.boundingRect(eye_cnt)
                        leftmost_eye_x = min(leftmost_eye_x, eye_x)
                body_width = leftmost_eye_x
                if body_width > 0:
                    body_region = (snake_x, snake_y, body_width, snake_h)
                    results["snake_body"].append(body_region)
                head_x = snake_x + leftmost_eye_x
                head_width = snake_w - leftmost_eye_x
                if head_width > 0:
                    head_region = (head_x, snake_y, head_width, snake_h)
                    results["snake_head"].append(head_region)
                if body_width > 0:
                    cv2.rectangle(
                        output,
                        (snake_x, snake_y),
                        (snake_x + body_width, snake_y + snake_h),
                        (255, 0, 0),
                        2,
                    )
                    cv2.putText(
                        output,
                        "Snake Body",
                        (snake_x, snake_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                if head_width > 0:
                    cv2.rectangle(
                        output,
                        (head_x, snake_y),
                        (head_x + head_width, snake_y + snake_h),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        output,
                        "Snake Head",
                        (head_x, snake_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
            else:
                body_width = (snake_w * 2) // 3
                head_width = snake_w - body_width
                body_region = (snake_x, snake_y, body_width, snake_h)
                results["snake_body"].append(body_region)
                head_x = snake_x + body_width
                head_region = (head_x, snake_y, head_width, snake_h)
                results["snake_head"].append(head_region)
                cv2.rectangle(
                    output,
                    (snake_x, snake_y),
                    (snake_x + body_width, snake_y + snake_h),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    output,
                    "Snake Body",
                    (snake_x, snake_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
                cv2.rectangle(
                    output,
                    (head_x, snake_y),
                    (head_x + head_width, snake_y + snake_h),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    output,
                    "Snake Head",
                    (head_x, snake_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

    # Detect Apple (Red)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_apple = cv2.bitwise_or(mask_red1, mask_red2)
    contours_apple, _ = cv2.findContours(
        mask_apple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours_apple:
        area = cv2.contourArea(cnt)
        if area > 30:
            x, y, w, h = cv2.boundingRect(cnt)
            results["apple"].append((x, y, w, h))
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                output,
                "Apple",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    return results, output, mask_snake
