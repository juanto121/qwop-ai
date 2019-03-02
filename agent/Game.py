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
        self.game_steps = 0

    def start(self):
        self.agent.start_game()
        self.game_steps = 0
        return self.execute_action('n')

    def reload(self):
        self.game_steps = 0
        self.agent.hard_reload()
        self.agent.start_game()
        return self.execute_action('n')

    def soft_reload(self):
        self.agent.reload()

    def execute_action(self, action):
        self.game_steps += 1
        # TODO: Optimize execution of actions (selenium is slow!)
        #self.agent.unpause()
        for char in action:
            getattr(self.agent, char)()
        shot = self.get_screen_shot()
        #self.agent.pause()
        #TODO: Move logic of limiting every game to max K steps
        done = self.is_done(shot)
        score = 0.0
        if done:
            distance_score = self.get_score()
            time_score = 0 # -((self.game_steps * 0.1)/(abs(distance_score)+1e5))
            score = distance_score + time_score
            self.agent.reload()
        return shot.astype(np.float).ravel(), score, done

    def is_done(self, shot):
        print(shot.shape)
        blueidx = shot[:, :] < 24
        notblueidx = shot[:, :] >= 24
        shot[blueidx] = 255
        shot[notblueidx] = 0
        np.savetxt('sample_shot', shot[15:20,66:])
        mask = np.array([[0,0,255],[0,255,255],[0,255,255],[0,255,255],[0,0,255],[0,0,255]])
        return np.array_equal(shot[15:21,66:], mask)

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
        return float(score)

    def get_screen_shot_timed(self):
        start = time.time()
        img = self.get_screen_shot()
        print(time.time() - start)
        return img

    def get_screen_shot(self, render = False):
        with mss.mss() as sct:
            shot = sct.grab({"top": 185, "left": 190, "width": 275, "height": 320})
            """
            TODO:
            TEST ONLY GRAYSCALE
            this processing might not be useful since the important data is in the difference between frames
           
            img = np.array(shot)
            img[:, :, 2] = 0
            img[:, :, 1] = 0
            blueidx = img[:, :, 0] < 24
            notblueidx = img[:, :, 0] >= 24
            img[blueidx] = 255
            img[notblueidx] = 0
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
             """
            img = np.array(shot)[:,:,0]
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            if render: self.render(img)
        return img

    def render(self, img):
        cv2.imshow('window', img)
        cv2.waitKey(1)
