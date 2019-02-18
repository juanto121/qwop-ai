from Agent import Agent
import logging
import time
from PIL import ImageGrab

class Game:

    def __init__(self):
        self.agent = Agent()

    def start(self):
        self.agent.reload()
        self.agent.start_game()

    def execute_action(self, action):
        for char in action:
            logging.debug(f'Executing action ${char}')
            getattr(self.agent, char)()

    def get_score(self):
        return int(1)

    def get_screen_shot(self):
        while(True):
            start = time.time()
            shot = ImageGrab.grab(bbox=(0,40,800,640))
            print(time.time() - start)
        return shot