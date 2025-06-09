import numpy as np
import cv2
# Directions
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
KEYS = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}

def improved_label_matrix_from_detection(frame, grid_xs, grid_ys):
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


def simulate_snake_after_apple(label_matrix, path, head):
    """Simulate the snake's body after eating the apple, return updated label_matrix and new head."""
    sim_matrix = label_matrix.copy()
    curr = head
    for move in path:
        curr = (curr[0] + move[0], curr[1] + move[1])
        sim_matrix[curr] = "s"
    return sim_matrix, curr


def get_next_safe_move(label_matrix, pos):
    """Return a safe direction from pos, or None if none."""
    n_rows, n_cols = label_matrix.shape
    for dr, dc in DIRECTIONS:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            if label_matrix[nr, nc] != "s":
                return (dr, dc)
    return None
