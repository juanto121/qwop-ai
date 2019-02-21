from Game import Game

class State:

    def __init__(self):
        self.game = Game()
        self.game.start()
        while True:
            observation, reward, done = self.game.execute_action("r")
            if done:
                print(reward)
                self.game.reload()


