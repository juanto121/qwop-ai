from Game import Game
import time

def main():
    game = Game()
    game.start()

    while True:
        observation, reward, done = game.execute_action("r")
        if done:
            print(f'reward {reward}')
            game.reload()

if __name__ == '__main__':
    main()