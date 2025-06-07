import cv2
import numpy as np

# Load the image directly
image = cv2.imread('Screenshot 2025-06-06 162938.png')
if image is None:
    print("Error: Could not load the screenshot image")
    exit()

output = image.copy()

# Convert to HSV color space for better color detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Store results
results = {
    'game_area': None,
    'snake_head': [],
    'snake_body': [],
    'apple': []
}

### --- 1. Detect Game Area (Checkerboard) ---
# Using LAB color space for better brightness detection
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l_channel = lab[:, :, 0]

# Find areas with higher brightness (the checkerboard area)
_, game_area_mask = cv2.threshold(l_channel, 120, 255, cv2.THRESH_BINARY)

# Clean up the mask
kernel = np.ones((5, 5), np.uint8)
game_area_mask = cv2.morphologyEx(game_area_mask, cv2.MORPH_CLOSE, kernel)
game_area_mask = cv2.morphologyEx(game_area_mask, cv2.MORPH_OPEN, kernel)

# Find the largest rectangular contour (should be the game area)
contours_game, _ = cv2.findContours(game_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

game_area_roi = None
if contours_game:
    # Get the largest contour
    largest_game_area = max(contours_game, key=cv2.contourArea)
    area = cv2.contourArea(largest_game_area)
    
    # Make sure it's large enough to be the game area
    if area > 10000:  # Adjust threshold as needed
        x, y, w, h = cv2.boundingRect(largest_game_area)
        results['game_area'] = (x, y, w, h)
        
        # Draw bounding box for game area
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 255), 3)
        cv2.putText(output, 'Game Area', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Create mask for game area only
        game_area_roi = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(game_area_roi, (x, y), (x+w, y+h), 255, -1)

### --- 2. Detect Snake Parts (Blue) ---
# Refined blue detection for snake
lower_blue = np.array([100, 120, 50])
upper_blue = np.array([130, 255, 255])
mask_snake = cv2.inRange(hsv, lower_blue, upper_blue)

# Only consider snake parts within the game area
if results['game_area'] and game_area_roi is not None:
    mask_snake = cv2.bitwise_and(mask_snake, game_area_roi)

contours_snake, _ = cv2.findContours(mask_snake, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours_snake:
    # Get the largest blue contour (should be the entire snake)
    largest_snake_contour = max(contours_snake, key=cv2.contourArea)
    snake_area = cv2.contourArea(largest_snake_contour)
    
    if snake_area > 100:
        # Get bounding rect of entire snake
        snake_x, snake_y, snake_w, snake_h = cv2.boundingRect(largest_snake_contour)
        
        # Create mask for just this snake contour
        snake_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(snake_mask, [largest_snake_contour], -1, 255, -1)
        
        # Detect white eyes within the snake area
        gray_roi = cv2.cvtColor(image[snake_y:snake_y+snake_h, snake_x:snake_x+snake_w], cv2.COLOR_BGR2GRAY)
        _, eye_thresh = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
        
        # Find eye contours
        eye_contours, _ = cv2.findContours(eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if eye_contours:
            # Find the leftmost eye to determine where eyes start
            leftmost_eye_x = snake_w  # Start from right end
            for eye_cnt in eye_contours:
                if cv2.contourArea(eye_cnt) > 10:  # Small circular areas (eyes)
                    eye_x, eye_y, eye_w, eye_h = cv2.boundingRect(eye_cnt)
                    leftmost_eye_x = min(leftmost_eye_x, eye_x)
            
            # Body goes from snake start to where eyes start
            body_width = leftmost_eye_x
            if body_width > 0:
                body_region = (snake_x, snake_y, body_width, snake_h)
                results['snake_body'].append(body_region)
            
            # Head goes from eyes start to snake end
            head_x = snake_x + leftmost_eye_x
            head_width = snake_w - leftmost_eye_x
            if head_width > 0:
                head_region = (head_x, snake_y, head_width, snake_h)
                results['snake_head'].append(head_region)
            
            # Draw the detections
            if body_width > 0:
                cv2.rectangle(output, (snake_x, snake_y), (snake_x + body_width, snake_y + snake_h), (255, 0, 0), 2)
                cv2.putText(output, 'Snake Body', (snake_x, snake_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if head_width > 0:
                cv2.rectangle(output, (head_x, snake_y), (head_x + head_width, snake_y + snake_h), (0, 0, 255), 2)
                cv2.putText(output, 'Snake Head', (head_x, snake_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        else:
            # Fallback: if no eyes detected, use right portion as head
            body_width = (snake_w * 2) // 3  # Use 2/3 of snake as body
            head_width = snake_w - body_width  # Remaining 1/3 as head
            
            body_region = (snake_x, snake_y, body_width, snake_h)
            results['snake_body'].append(body_region)
            
            head_x = snake_x + body_width
            head_region = (head_x, snake_y, head_width, snake_h)
            results['snake_head'].append(head_region)
            
            # Draw the detections
            cv2.rectangle(output, (snake_x, snake_y), (snake_x + body_width, snake_y + snake_h), (255, 0, 0), 2)
            cv2.putText(output, 'Snake Body', (snake_x, snake_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.rectangle(output, (head_x, snake_y), (head_x + head_width, snake_y + snake_h), (0, 0, 255), 2)
            cv2.putText(output, 'Snake Head', (head_x, snake_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

### --- 3. Detect Apple (Red) ---
# Refined red detection for apple
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_apple = cv2.bitwise_or(mask_red1, mask_red2)

# Only consider apples within the game area
if results['game_area'] and game_area_roi is not None:
    mask_apple = cv2.bitwise_and(mask_apple, game_area_roi)

contours_apple, _ = cv2.findContours(mask_apple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours_apple:
    area = cv2.contourArea(cnt)
    if area > 30:  # Minimum area for apple
        x, y, w, h = cv2.boundingRect(cnt)
        results['apple'].append((x, y, w, h))
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output, 'Apple', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

### --- 4. Create individual masks ---
# Create separate masks for each component
mask_game_area = np.zeros(image.shape[:2], dtype=np.uint8)
mask_snake_head = np.zeros(image.shape[:2], dtype=np.uint8)
mask_snake_body = np.zeros(image.shape[:2], dtype=np.uint8)
mask_apple_final = np.zeros(image.shape[:2], dtype=np.uint8)

# Fill game area mask
if results['game_area']:
    x, y, w, h = results['game_area']
    cv2.rectangle(mask_game_area, (x, y), (x+w, y+h), 255, -1)

# Fill snake head mask
for x, y, w, h in results['snake_head']:
    cv2.rectangle(mask_snake_head, (x, y), (x+w, y+h), 255, -1)

# Fill snake body mask
for x, y, w, h in results['snake_body']:
    cv2.rectangle(mask_snake_body, (x, y), (x+w, y+h), 255, -1)

# Fill apple mask
for x, y, w, h in results['apple']:
    cv2.rectangle(mask_apple_final, (x, y), (x+w, y+h), 255, -1)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Game Area Mask', mask_game_area)
cv2.imshow('Snake Head Mask', mask_snake_head)
cv2.imshow('Snake Body Mask', mask_snake_body)
cv2.imshow('Apple Mask', mask_apple_final)
cv2.imshow('All Detections', output)

# Print results
print("Detection Results:")
print(f"Game Area: {results['game_area']}")
print(f"Snake Head: {results['snake_head']}")
print(f"Snake Body: {results['snake_body']}")
print(f"Apple: {results['apple']}")

cv2.waitKey(0)
cv2.destroyAllWindows()