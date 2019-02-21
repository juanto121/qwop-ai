from Game import Game

class State:

    def __init__(self):
        self.game = Game()
        self.game.start()
        #self.game.get_screen_shot()
        self.game.get_screen_shot_timed()

