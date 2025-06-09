import numpy as np
import cv2


def adjust_roi(roi, padding_ratio=0.03):
    x, y, w, h = roi
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    new_x = x + pad_x
    new_y = y + pad_y
    new_w = w - 2 * pad_x
    new_h = h - 2 * pad_y
    return (new_x, new_y, new_w, new_h)


def roi_to_grid(roi, n_cols=17, n_rows=15):
    _, _, w, h = roi
    grid_xs = np.linspace(0, w, n_cols + 1, dtype=int)
    grid_ys = np.linspace(0, h, n_rows + 1, dtype=int)
    return grid_xs, grid_ys


def point_to_cell(px, py, grid_xs, grid_ys, roi):
    roi_x, roi_y, _, _ = roi
    px_roi = px - roi_x
    py_roi = py - roi_y
    col = np.searchsorted(grid_xs, px_roi, side="right") - 1
    row = np.searchsorted(grid_ys, py_roi, side="right") - 1
    col = max(0, min(col, len(grid_xs) - 2))
    row = max(0, min(row, len(grid_ys) - 2))
    return row, col


def bbox_center(bbox):
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


def bbox_cells(bbox, grid_xs, grid_ys, roi):
    x, y, w, h = bbox
    cells = set()
    for px in np.linspace(x, x + w - 1, num=3, dtype=int):
        for py in np.linspace(y, y + h - 1, num=3, dtype=int):
            row, col = point_to_cell(px, py, grid_xs, grid_ys, roi)
            cells.add((row, col))
    return list(cells)


def get_body_spine_points(body_bbox, mask_snake, min_length=5):
    x, y, w, h = body_bbox
    body_mask = mask_snake[y : y + h, x : x + w]
    body_mask = cv2.threshold(body_mask, 127, 255, cv2.THRESH_BINARY)[1]
    skeleton = np.zeros(body_mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = np.zeros(body_mask.shape, np.uint8)
    done = False
    while not done:
        eroded = cv2.erode(body_mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(body_mask, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        body_mask = eroded.copy()
        done = cv2.countNonZero(body_mask) == 0
    ys, xs = np.where(skeleton > 0)
    points = [(x + int(px), y + int(py)) for py, px in zip(ys, xs)]
    if len(points) < min_length:
        return []
    return points


def print_game_matrix(matrix):
    # Accept NumPy arrays as well
    symbols = {
        0: ".",
        1: "S",
        2: "A",
        3: "#",
        4: "H",
        ".": ".",
        "s": "S",
        "a": "A",
        "#": "#",
        "H": "H",
    }
    for row in matrix:
        print(" ".join(symbols.get(cell, "?") for cell in row))
    print()
