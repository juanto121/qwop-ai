from Agent import Agent
import numpy as np
import logging
import time
import pytesseract
import cv2
import mss
import re

class Game:

    def __init__(self):
        self.agent = Agent()

    def start(self):
        self.agent.reload()
        self.agent.start_game()

    def reload(self):
        self.agent.space()

    def execute_action(self, action):
        # TODO: Optimize execution of actions (selenium is slow!)
        for char in action:
            getattr(self.agent, char)()
        shot = self.get_screen_shot()
        done = self.is_done(shot)
        score = 0
        if done:
            score = self.get_score()
        return (shot, score, self.is_done(shot))

    def is_done(self, shot):
        mask = np.array([[0,0,191],[0,128,255],[0,255,255],[0,255,255],[0,255,255],[0,64,255],[0,0,64]])
        return np.array_equal(shot[18:25,66:], mask)

    def get_score(self):
        with mss.mss() as sct:
            shot = sct.grab({"top": 155, "left": 140, "width": 350, "height": 70})
            img = ~(np.array(shot)[:,:,0]) #removes rgb and inverts colors
            img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1] #threshold to remove color artifacts and leave it black and white
            score = pytesseract.image_to_string(image=img)
            digits_rgx = re.compile("-?[0-9]+.?[0-9]")
            result = digits_rgx.findall(score)
            if len(result) > 0:
                score = result[0]
            else:
                score = 0
        return score

    def get_screen_shot_timed(self):
        while True:
            start = time.time()
            img = self.get_screen_shot()
            cv2.imshow('window', img)
            cv2.waitKey(1)
            print(time.time() - start)
        return img

    def get_screen_shot(self):
        with mss.mss() as sct:
            """
            TODO:
            TEST ONLY GRAYSCALE
            this processing might not be useful since the important data is in the difference between frames
            """
            shot = sct.grab({"top": 170, "left": 190, "width": 275, "height": 320})
            img = np.array(shot)
            img[:, :, 2] = 0
            img[:, :, 1] = 0
            blueidx = img[:, :, 0] < 24
            notblueidx = img[:, :, 0] >= 24
            img[blueidx] = 255
            img[notblueidx] = 0
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        return img
