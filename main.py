import webbrowser
import pyautogui
import time
import cv2
import mss
import numpy as np
pyautogui.FAILSAFE = False


# Step 1: Open game in real browser
webbrowser.open("https://www.google.com/search?q=play+snake")
time.sleep(5)  # Wait for game to load

# Step 2: Locate Play Button on screen
play_btn = pyautogui.locateOnScreen("assets/playbtn.png", confidence=0.9)
pyautogui.click(pyautogui.center(play_btn))
print("Clicked Play button!")
time.sleep(3)  # Wait for second start screen

startplay = pyautogui.locateOnScreen("./assets/start.png", confidence=0.9)
pyautogui.click(pyautogui.center(startplay))

print("StartedPlaying!")


def find_game_area():
    # Take screenshot using pyautogui (we use this just once)
    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define green range (tweak these values based on your screen)
    lower_green = np.array([25, 100, 50])
    upper_green = np.array([75, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    game_rect = (0, 0, 1200, 800)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        area = w * h

        # Conditions to filter likely square game region
        if 0.9 < aspect_ratio < 1.1 and area > 30000:
            game_rect = (x, y, w, h)
            break

    return game_rect


def capture_game_loop(region):
    x, y, w, h = region
    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": w, "height": h}
        frame_count = 0

        # Create window and set to small size
        cv2.namedWindow("Game Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Game Feed", 200, 200)  # Set window size to 200x200

        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Show the game ROI in a window
            cv2.imshow("Game Feed", frame)

            # Exit with ESC key
            if cv2.waitKey(1) == ord("q"):
                print("framecount: ", frame_count)
                break

            frame_count += 1

        cv2.destroyAllWindows()


time.sleep(2)  # Give user time to position the window

roi = find_game_area()
if roi:
    print(f"Game area detected at: {roi}")
    capture_game_loop(roi)
else:
    print("Could not detect game area. Make sure it's visible and green.")
