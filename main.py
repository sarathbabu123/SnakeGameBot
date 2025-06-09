import time
import mss
import keyboard
import cv2
import numpy as np
from game_launcher import launch_game
from vision import find_game_area
from matrix_utils import adjust_roi, roi_to_grid
from snake_utils import (
    improved_label_matrix_from_detection,
    find_apple,
    bfs_path,
    get_safe_dirs,
    is_in_bounds,
    simulate_snake_after_apple,
    get_next_safe_move,
)

# Directions
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
KEYS = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}


def label_matrix_from_detection(frame, grid_xs, grid_ys):
    n_rows = len(grid_ys) - 1
    n_cols = len(grid_xs) - 1
    label_matrix = np.full((n_rows, n_cols), ".", dtype="<U1")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_snake = cv2.inRange(hsv, (100, 120, 50), (130, 255, 255))
    # Improved apple detection (handle different red shades)
    mask_apple1 = cv2.inRange(hsv, (0, 150, 50), (10, 255, 255))  # Bright red
    mask_apple2 = cv2.inRange(hsv, (170, 150, 50), (180, 255, 255))  # Dark red
    mask_apple = cv2.bitwise_or(mask_apple1, mask_apple2)
    mask_eye = cv2.inRange(hsv, (0, 0, 200), (180, 25, 255))  # bright white range

    for row in range(n_rows):
        for col in range(n_cols):
            x1, y1 = grid_xs[col], grid_ys[row]
            x2, y2 = grid_xs[col + 1], grid_ys[row + 1]
            cell_snake = mask_snake[y1:y2, x1:x2]
            cell_apple = mask_apple[y1:y2, x1:x2]
            area = (y2 - y1) * (x2 - x1)
            if np.sum(cell_snake > 0) > area * 0.22:
                label_matrix[row, col] = "s"
            elif np.sum(cell_apple > 0) > area * 0.12:
                label_matrix[row, col] = "a"

    # Enhanced eye detection with contour filtering and grid mapping
    eye_positions = []
    contours, _ = cv2.findContours(mask_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if 2 <= cv2.contourArea(contour) <= 10:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            col_idx = np.searchsorted(grid_xs, center_x) - 1
            row_idx = np.searchsorted(grid_ys, center_y) - 1
            if 0 <= row_idx < n_rows and 0 <= col_idx < n_cols:
                eye_positions.append((row_idx, col_idx))
    return label_matrix, eye_positions


def find_snake_head(
    label_matrix,
    eye_positions,
    last_head=None,
    last_dir=None,
    prev_head=None,
    head_history=None,
    grid_xs=None,
    grid_ys=None,
    frame=None,
):
    body = [
        (r, c)
        for r in range(label_matrix.shape[0])
        for c in range(label_matrix.shape[1])
        if label_matrix[r, c] == "s"
    ]
    if not body:
        return None

    # 1. Find tips (cells with only 1 neighbor)
    neighbor_counts = {}
    for r, c in body:
        count = 0
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < label_matrix.shape[0]
                and 0 <= nc < label_matrix.shape[1]
                and label_matrix[nr, nc] == "s"
            ):
                count += 1
        neighbor_counts[(r, c)] = count
    tips = [pos for pos, count in neighbor_counts.items() if count == 1]

    # 2. Priority 1: Tips with eyes (most reliable)
    eye_tips = [pos for pos in tips if pos in eye_positions]
    if eye_tips:
        if len(eye_tips) > 1 and head_history and len(head_history) > 0:
            return min(
                eye_tips,
                key=lambda p: abs(p[0] - head_history[-1][0])
                + abs(p[1] - head_history[-1][1]),
            )
        return eye_tips[0]

    # 3. Priority 2: Direction-based prediction
    if last_dir and last_head and head_history:
        predicted_head = (last_head[0] + last_dir[0], last_head[1] + last_dir[1])
        if predicted_head in tips:
            return predicted_head

    # 4. Priority 3: Dark blue detection (head vs lighter neck)
    if frame is not None and grid_xs is not None and grid_ys is not None:
        dark_blue_tips = []
        for tip in tips:
            r, c = tip
            x1, y1 = grid_xs[c], grid_ys[r]
            x2, y2 = grid_xs[c + 1], grid_ys[r + 1]
            cell = frame[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            blue_intensity = np.mean(cell[:, :, 0])  # Blue channel
            green_intensity = np.mean(cell[:, :, 1])  # Green channel
            if blue_intensity > 100 and green_intensity < 80:
                dark_blue_tips.append(tip)
        if dark_blue_tips:
            return dark_blue_tips[0]

    # 5. Fallback strategies (existing logic)
    if len(tips) == 2 and head_history and len(head_history) > 0:
        recent_head = head_history[-1]
        return min(
            tips,
            key=lambda pos: abs(pos[0] - recent_head[0]) + abs(pos[1] - recent_head[1]),
        )
    if len(tips) == 2:
        centroid_r = sum(r for r, _ in body) / len(body)
        centroid_c = sum(c for _, c in body) / len(body)
        return max(
            tips,
            key=lambda pos: (pos[0] - centroid_r) ** 2 + (pos[1] - centroid_c) ** 2,
        )
    if len(tips) == 1:
        return tips[0]
    if tips:
        centroid_r = sum(r for r, _ in body) / len(body)
        centroid_c = sum(c for _, c in body) / len(body)
        return max(
            tips,
            key=lambda pos: (pos[0] - centroid_r) ** 2 + (pos[1] - centroid_c) ** 2,
        )
    centroid_r = sum(r for r, _ in body) / len(body)
    centroid_c = sum(c for _, c in body) / len(body)
    return max(
        body,
        key=lambda pos: (pos[0] - centroid_r) ** 2 + (pos[1] - centroid_c) ** 2,
    )


def capture_game_loop(region):
    grid_xs, grid_ys = roi_to_grid((0, 0, region[2], region[3]), 17, 15)
    monitor = {
        "top": region[1],
        "left": region[0],
        "width": region[2],
        "height": region[3],
    }

    print("Press SPACE to start bot...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("snake_run.mp4", fourcc, 30.0, (region[2], region[3]))
    prev_key = None
    last_dir = RIGHT
    last_head = None
    prev_head = None
    planned_path = []
    path_cells = []
    head_history = []
    while True:
        while not keyboard.is_pressed("space"):
            time.sleep(0.0001)
        print("Bot started. (Press Q in window to stop)")
        last_dir = RIGHT
        last_head = None
        prev_key = None
        prev_head = None
        planned_path = []
        path_cells = []
        keyboard.press_and_release(KEYS[RIGHT])
        with mss.mss() as sct:
            stuck_counter = 0
            while True:
                frame = np.asarray(sct.grab(monitor))
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                label_matrix, eye_positions = improved_label_matrix_from_detection(
                    frame, grid_xs, grid_ys
                )
                frame_np = frame

                # --- HEAD DETECTION LOGIC (work here) ---
                head = find_snake_head(
                    label_matrix,
                    eye_positions,
                    last_head,
                    last_dir,
                    prev_head,
                    head_history,
                    grid_xs,
                    grid_ys,
                    frame,
                )
                if head is not None:
                    head_history.append(head)
                    if len(head_history) > 5:
                        head_history.pop(0)
                apple = find_apple(label_matrix)
                move = None

                recalc_path = (
                    not planned_path
                    or not head
                    or not apple
                    or head != path_cells[0]
                    or label_matrix[apple] != "a"
                )

                if recalc_path and head and apple:
                    path = bfs_path(label_matrix, head, apple)
                    valid = True
                    curr = head
                    for m in path:
                        next_pos = (curr[0] + m[0], curr[1] + m[1])
                        if not is_in_bounds(next_pos, label_matrix) or (
                            label_matrix[next_pos] == "s" and next_pos != apple
                        ):
                            valid = False
                            break
                        curr = next_pos
                    if valid and path:
                        planned_path = path
                        path_cells = [head]
                        curr = head
                        for m in path:
                            curr = (curr[0] + m[0], curr[1] + m[1])
                            path_cells.append(curr)
                    else:
                        planned_path = []
                        path_cells = []

                if planned_path and head == path_cells[0]:
                    move = planned_path[0]
                    if len(planned_path) > 1:
                        curr_dir = planned_path[0]
                        next_dir = planned_path[1]
                        if curr_dir != next_dir:
                            move = next_dir
                    next_pos = (head[0] + move[0], head[1] + move[1])
                    if not is_in_bounds(next_pos, label_matrix):
                        print("Next move would hit wall. Stopping.")
                        prev_key = None
                        break
                    planned_path = planned_path[1:]
                    path_cells = path_cells[1:]
                else:
                    safe_dirs = (
                        get_safe_dirs(label_matrix, head) if head is not None else []
                    )
                    safe_dirs = [
                        d
                        for d in safe_dirs
                        if is_in_bounds((head[0] + d[0], head[1] + d[1]), label_matrix)
                    ]
                    if last_dir in safe_dirs:
                        move = last_dir
                    elif safe_dirs:
                        move = safe_dirs[0]

                # --- Instant key event ---
                if move:
                    key = KEYS[move]
                    if prev_key != key:
                        keyboard.press_and_release(key)
                    prev_key = key
                    last_dir = move
                    last_head = head
                else:
                    prev_key = None
                    print("No safe directions.")
                    break

                if prev_head == head:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                prev_head = head
                if stuck_counter > 10:
                    print("Snake stopped moving. Game over.")
                    break

                # --- Drawing grid lines and path before writing video ---
                for x in grid_xs:
                    cv2.line(frame_np, (x, 0), (x, region[3]), (200, 200, 200), 1)
                for y in grid_ys:
                    cv2.line(frame_np, (0, y), (region[2], y), (200, 200, 200), 1)
                # Draw planned path as a yellow polyline
                if path_cells and len(path_cells) > 1:
                    pts = []
                    for r, c in path_cells:
                        cx = (grid_xs[c] + grid_xs[c + 1]) // 2
                        cy = (grid_ys[r] + grid_ys[r + 1]) // 2
                        pts.append((cx, cy))
                    cv2.polylines(
                        frame_np,
                        [np.array(pts, dtype=np.int32)],
                        False,
                        (0, 255, 255),
                        2,
                    )
                # --- Draw bounding box over head and body ---
                # Find all snake cells
                snake_cells = [
                    (r, c)
                    for r in range(label_matrix.shape[0])
                    for c in range(label_matrix.shape[1])
                    if label_matrix[r, c] == "s"
                ]
                if snake_cells:
                    rows, cols = zip(*snake_cells)
                    min_r, max_r = min(rows), max(rows)
                    min_c, max_c = min(cols), max(cols)
                    x1, y1 = grid_xs[min_c], grid_ys[min_r]
                    x2, y2 = grid_xs[max_c + 1], grid_ys[max_r + 1]
                    cv2.rectangle(frame_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame_np,
                        "Body",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                # Draw bounding box over head
                if path_cells:
                    head_r, head_c = path_cells[0]
                    x1, y1 = grid_xs[head_c], grid_ys[head_r]
                    x2, y2 = grid_xs[head_c + 1], grid_ys[head_r + 1]
                    cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame_np,
                        "Head",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                # --- Debug Visualization: draw head and eyes ---
                if head:
                    r, c = head
                    x1, y1 = grid_xs[c], grid_ys[r]
                    x2, y2 = grid_xs[c + 1], grid_ys[r + 1]
                    cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    for eye_r, eye_c in eye_positions:
                        if (eye_r, eye_c) == head:
                            ex = (grid_xs[eye_c] + grid_xs[eye_c + 1]) // 2
                            ey = (grid_ys[eye_r] + grid_ys[eye_r + 1]) // 2
                            cv2.circle(frame_np, (ex, ey), 3, (0, 255, 255), -1)

                out.write(frame_np)
            print("Bot stopped. Press SPACE to start again.")
    out.release()


if __name__ == "__main__":
    launch_game()
    time.sleep(2)
    game_area_img, roi = find_game_area()
    if roi:
        roi = adjust_roi(roi, padding_ratio=0.04)
        print(f"Game area (with padding) at: {roi}")
        capture_game_loop(roi)
    else:
        print("Could not detect game area. Make sure it's visible and green.")
