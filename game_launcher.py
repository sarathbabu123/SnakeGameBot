import webbrowser
import pyautogui
import time

pyautogui.FAILSAFE = False


def launch_game():
    """
    Opens the Google Snake game in a browser and starts the game.
    """
    webbrowser.open("https://www.google.com/search?q=play+snake")
    time.sleep(2)  # Wait for game to load

    play_btn = pyautogui.locateOnScreen("assets/playbtn.png", confidence=0.9)
    pyautogui.click(pyautogui.center(play_btn))
    print("Clicked Play button!")
    time.sleep(3)  # Wait for second start screen

    startplay = pyautogui.locateOnScreen("./assets/start.png", confidence=0.9)
    pyautogui.click(pyautogui.center(startplay))
    print("Started Playing!")
