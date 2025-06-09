import time
import mss
import keyboard
import cv2
import numpy as np
from game_launcher import launch_game
from vision import find_game_area
from matrix_utils import adjust_roi, roi_to_grid

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
    mask_apple1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    mask_apple2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    mask_apple = cv2.bitwise_or(mask_apple1, mask_apple2)
    mask_eye = cv2.inRange(hsv, (0, 0, 200), (180, 25, 255))
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
    # Eye detection: vectorized
    eye_positions = [
        (row, col)
        for row in range(n_rows)
        for col in range(n_cols)
        if np.sum(
            mask_eye[grid_ys[row] : grid_ys[row + 1], grid_xs[col] : grid_xs[col + 1]]
        )
        > (grid_xs[col + 1] - grid_xs[col]) * (grid_ys[row + 1] - grid_ys[row]) * 0.02
    ]
    return label_matrix, eye_positions


def find_snake_head(
    label_matrix, eye_positions, last_head=None, last_dir=None, prev_head=None
):
    # Vectorized tip/eye/motion logic
    body = [
        (r, c)
        for r in range(label_matrix.shape[0])
        for c in range(label_matrix.shape[1])
        if label_matrix[r, c] == "s"
    ]
    if not body:
        return None
    neighbor_counts = {}
    for r, c in body:
        count = sum((r + dr, c + dc) in body for dr, dc in DIRECTIONS)
        neighbor_counts[(r, c)] = count
    tips = [pos for pos, count in neighbor_counts.items() if count == 1]
    eye_tips = [pos for pos in tips if pos in eye_positions]
    # 1. Eye tip
    if len(eye_tips) == 1:
        return eye_tips[0]
    # 2. Two eye tips: use motion or last_head
    if len(eye_tips) == 2:
        if last_head and prev_head:
            dr = last_head[0] - prev_head[0]
            dc = last_head[1] - prev_head[1]
            predicted = (last_head[0] + dr, last_head[1] + dc)
            if predicted in eye_tips:
                return predicted
        if last_head and last_head in eye_tips:
            return last_head
        return max(
            eye_tips,
            key=lambda pos: min(
                pos[0],
                pos[1],
                label_matrix.shape[0] - 1 - pos[0],
                label_matrix.shape[1] - 1 - pos[1],
            ),
        )
    # 3. Any eye cell
    if eye_positions:
        return eye_positions[0]
    # 4. Fallback: tip logic
    if len(tips) == 1:
        return tips[0]
    if len(tips) == 2:
        if last_head and last_head in tips:
            other_tip = tips[0] if tips[1] == last_head else tips[1]
            if last_dir:
                predicted = (last_head[0] + last_dir[0], last_head[1] + last_dir[1])
                if predicted == other_tip:
                    return other_tip
            return last_head
        elif last_head:
            return min(
                tips,
                key=lambda pos: abs(pos[0] - last_head[0]) + abs(pos[1] - last_head[1]),
            )
        else:
            return max(
                tips,
                key=lambda pos: min(
                    pos[0],
                    pos[1],
                    label_matrix.shape[0] - 1 - pos[0],
                    label_matrix.shape[1] - 1 - pos[1],
                ),
            )
    if last_head:
        possible_heads = [pos for pos in body if neighbor_counts.get(pos, 0) <= 2]
        predicted = (
            last_head[0] + (last_dir[0] if last_dir else 0),
            last_head[1] + (last_dir[1] if last_dir else 0),
        )
        if predicted in possible_heads:
            return predicted
        return min(
            possible_heads,
            key=lambda pos: abs(pos[0] - last_head[0]) + abs(pos[1] - last_head[1]),
        )
    return min(body)


def find_apple(label_matrix):
    for r in range(label_matrix.shape[0]):
        for c in range(label_matrix.shape[1]):
            if label_matrix[r, c] == "a":
                return (r, c)
    return None


def bfs_path(label_matrix, start, goal):
    # BFS to find the shortest safe path (never through body or wall)
    from collections import deque

    n_rows, n_cols = label_matrix.shape
    visited = np.zeros_like(label_matrix, dtype=bool)
    prev = {}
    queue = deque()
    queue.append(start)
    visited[start] = True
    found = False
    while queue:
        curr = queue.popleft()
        if curr == goal:
            found = True
            break
        for dr, dc in DIRECTIONS:
            nr, nc = curr[0] + dr, curr[1] + dc
            # Out-of-bounds is treated as wall
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                if not visited[nr, nc] and (
                    label_matrix[nr, nc] != "s" or (nr, nc) == goal
                ):
                    visited[nr, nc] = True
                    prev[(nr, nc)] = (curr, (dr, dc))
                    queue.append((nr, nc))
    if not found:
        return []
    # Reconstruct path
    path = []
    curr = goal
    while curr != start:
        curr, move = prev[curr]
        path.append(move)
    path.reverse()
    return path


def get_safe_dirs(label_matrix, pos):
    """Return list of safe directions from pos (not out-of-bounds or into body)."""
    n_rows, n_cols = label_matrix.shape
    safe_dirs = []
    for dr, dc in DIRECTIONS:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            if label_matrix[nr, nc] != "s":
                safe_dirs.append((dr, dc))
    return safe_dirs


def is_in_bounds(pos, label_matrix):
    r, c = pos
    return 0 <= r < label_matrix.shape[0] and 0 <= c < label_matrix.shape[1]


def get_next_safe_after_apple(label_matrix, pos):
    """Return the safest direction from pos after apple is eaten (not out-of-bounds or into body)."""
    n_rows, n_cols = label_matrix.shape
    for dr, dc in DIRECTIONS:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            if label_matrix[nr, nc] != "s":
                return (dr, dc)
    return None


def capture_game_loop(region):
    _, _, w, h = region
    grid_xs, grid_ys = roi_to_grid((0, 0, w, h), 17, 15)
    monitor = {"top": region[1], "left": region[0], "width": w, "height": h}

    print("Press SPACE to start bot...")
    prev_key = None
    last_dir = RIGHT
    last_head = None
    prev_head = None
    planned_path = []
    path_cells = []
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
            next_move_after_apple = None
            apple_pos = None
            while True:
                frame = np.asarray(sct.grab(monitor))
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                label_matrix, eye_positions = label_matrix_from_detection(
                    frame, grid_xs, grid_ys
                )
                head = find_snake_head(
                    label_matrix, eye_positions, last_head, last_dir, prev_head
                )
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
                        # Predict next safe move after apple
                        apple_pos = path_cells[-1]
                        # Simulate snake with apple eaten
                        sim_matrix = label_matrix.copy()
                        for cell in path_cells:
                            sim_matrix[cell] = "s"
                        next_move_after_apple = get_next_safe_after_apple(
                            sim_matrix, apple_pos
                        )
                    else:
                        planned_path = []
                        path_cells = []
                        next_move_after_apple = None
                        apple_pos = None

                # If head is at apple, immediately turn to next safe direction
                if (
                    apple_pos is not None
                    and head == apple_pos
                    and next_move_after_apple
                ):
                    move = next_move_after_apple
                    planned_path = []
                    path_cells = []
                    next_move_after_apple = None
                    apple_pos = None
                elif planned_path and head == path_cells[0]:
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

                # Drawing/video can be commented for max speed
                # out.write(frame)
            print("Bot stopped. Press SPACE to start again.")


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
