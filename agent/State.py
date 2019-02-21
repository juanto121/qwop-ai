from Game import Game

class State:

    def __init__(self):
        self.game = Game()
        self.game.start()
        self.game.execute_action('q')
        self.game.get_screen_shot()
        while True:
            score = self.game.get_score()
            if(score != 0):
                print(score)
        #self.game.get_screen_shot_timed()

