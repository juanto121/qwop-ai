from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np
import time
from pynput.keyboard import Key, Controller

class Agent:

    def __init__(self):
        chrome_opts = Options()

        # REFER TO https://github.com/zalando/zalenium/issues/497 for enabling flash in docker
        chrome_opts.add_argument('--disable-features=EnableEphemeralFlashPermission')
        chrome_opts.add_argument('--disable-infobars')
        chrome_opts.add_argument("--ppapi-flash-version=32.0.0.101")
        chrome_opts.add_argument("--ppapi-flash-path=/usr/lib/pepperflashplugin-nonfree/libpepflashplayer.so")


        chrome_prefs = {"profile.default_content_setting_values.plugins": 1,
                        "profile.content_settings.plugin_whitelist.adobe-flash-player": 1,
                        "profile.content_settings.exceptions.plugins.*,*.per_resource.adobe-flash-player": 1,
                        "PluginsAllowedForUrls": "localhost"}
        chrome_opts.add_experimental_option('prefs', chrome_prefs)
        self.driver = webdriver.Chrome(executable_path = './chrome-driver/chromedriver', chrome_options = chrome_opts)
        self.driver.get('localhost:3000')
        self.canvas = self.driver.find_element_by_id('game-canvas')
        self.canvas_size = {"w":600, "h":400}
        self.is_paused = False
        self.keyboard = Controller()

    # no-op
    def n(self):
        return

    def o(self):
        self.keyboard.press('o')
        time.sleep(0.08)
        self.keyboard.release('o')

    def q(self):
        self.keyboard.press('q')
        time.sleep(0.08)
        self.keyboard.release('q')

    def w(self):
        self.keyboard.press('w')
        time.sleep(0.08)
        self.keyboard.release('w')

    def p(self):
        self.keyboard.press('p')
        time.sleep(0.08)
        self.keyboard.release('p')

    def r(self):
        choice = np.random.choice(['q','w','o','p'])
        self.keyboard.press(choice)
        time.sleep(0.08)
        self.keyboard.release(choice)

    def space(self):
        self.keyboard.press(Key.space)
        time.sleep(0.08)
        self.keyboard.release(Key.space)

    def start_game(self):
        self.canvas = self.driver.find_element_by_id('game-canvas')
        self.canvas.click()

    def click_tutorial(self):
        ac = ActionChains(self.driver)
        ac.move_to_element(self.canvas)
        ac.move_by_offset(-self.canvas_size["w"]/2+10, -self.canvas_size["h"]/2+20)
        ac.click()
        ac.perform()

    def pause(self):
        self.click_tutorial()

    def unpause(self):
        self.click_tutorial()

    def hard_reload(self):
        self.keyboard.press(Key.f5)
        time.sleep(0.08)
        self.keyboard.release(Key.f5)

    def reload(self):
        self.keyboard.press('r')
        time.sleep(0.08)
        self.keyboard.release('r')

    def screen_shot(self):
        return self.canvas.screenshot_as_base64
