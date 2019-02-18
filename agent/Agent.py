from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

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

    def o(self):
        self.canvas.send_keys('o')

    def q(self):
        self.canvas.send_keys('q')

    def w(self):
        self.canvas.send_keys('q')

    def p(self):
        self.canvas.send_keys('p')

    def space(self):
        self.canvas.send_keys(Keys.SPACE)

    def start_game(self):
        self.canvas.click()

    def reload(self):
        self.canvas.send_keys(Keys.F5)

    def screen_shot(self):
        return self.canvas.screenshot_as_base64
