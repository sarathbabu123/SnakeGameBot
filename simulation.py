import time
import random

# Constants for board values
EMPTY = 0
SNAKE = 1
APPLE = 2
WALL = 3

# Directions
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Board size
ROWS, COLS = 10, 10

def create_board():
    board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
    for r in range(ROWS):
        board[r][0] = board[r][-1] = WALL
    for c in range(COLS):
        board[0][c] = board[-1][c] = WALL
    return board

def place_apple(board, snake):
    while True:
        r, c = random.randint(1, ROWS-2), random.randint(1, COLS-2)
        if board[r][c] == EMPTY:
            board[r][c] = APPLE
            return (r, c)

def print_board(board):
    symbols = {EMPTY: '.', SNAKE: 'S', APPLE: 'A', WALL: '#'}
    for row in board:
        print(' '.join(symbols[cell] for cell in row))
    print()

def move_snake(board, snake, direction):
    head = snake[0]
    new_head = (head[0] + direction[0], head[1] + direction[1])
    r, c = new_head

    if board[r][c] in (WALL, SNAKE):
        return False  # Game over

    if board[r][c] == APPLE:
        snake.insert(0, new_head)  # Grow
        board[r][c] = SNAKE
        return 'ate'

    # Normal move
    snake.insert(0, new_head)
    board[r][c] = SNAKE
    tail = snake.pop()
    board[tail[0]][tail[1]] = EMPTY
    return True

def simulate():
    board = create_board()
    snake = [(5, 5)]
    board[5][5] = SNAKE
    apple_pos = place_apple(board, snake)
    direction = RIGHT
    head = snake[0]
    new_head = (head[0] + direction[0], head[1] + direction[1])
    snake.insert(0, new_head)

    while True:
        print_board(board)
        time.sleep(0.3)

        # Pick random safe direction
        safe_moves = []
        for d in DIRECTIONS:
            r, c = snake[0][0] + d[0], snake[0][1] + d[1]
            if board[r][c] in (EMPTY, APPLE):
                safe_moves.append(d)

        if not safe_moves:
            print("Game Over! No safe moves.")
            break

        direction = random.choice(safe_moves)
        result = move_snake(board, snake, direction)

        if result == False:
            print("Game Over! Collision.")
            break
        elif result == 'ate':
            apple_pos = place_apple(board, snake)

if __name__ == '__main__':
    simulate()
