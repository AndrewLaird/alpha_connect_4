from alpha_model import AlphaModel
import Connect4Logic
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss
from torch.nn.modules.activation import ReLU, Tanh
import torch.optim as optim
import sys
from os import path
import signal
from tqdm import tqdm
import sys
from collections import namedtuple
from multiprocessing import Pool
from multiprocessing import cpu_count

class fixed_size_list():
    def __init__(self,size):
        # keep same size
        # but able to add infinitely
        self.size = size
        self.buffer = []
        self.at_capacity = False
        self.index = 0

    def extend(self,items):
        if(self.at_capacity):
            for item in items:
                self.buffer[self.index] = item
                self.index = (self.index + 1 ) % self.size
        else:
            # add until we are at capacity
            if(len(items) + self.index < self.size):
                self.buffer.extend(items)
            else:
                # split it into two parts
                spliting_point  = self.size - self.index
                extend_data = items[:splitting_point]
                rotating_buffer_data = items[splitting_point:]

                self.extend(extend_data)

                self.at_capacity = True
                
                self.extend(rotating_buffer_data)
    def get_list(self):
        return self.buffer

def get_model_action(model,board,turn):
	return model.get_action_probs(board,turn,num_rollouts=25)

def pit_parallel(old_model,new_model,number_of_games):
    _,old_first_winners = run_game_parllel(old_model,new_model,num_games//2,temp=0.0,num_rollouts=20)
    _,new_first_winners = run_game_parllel(new_model,old_model,num_games//2,temp=0.0,num_rollouts=20)

    total_winners = {0:0,1:0,2:0}
    for winners in [old_first_winners,new_first_winners]:
        total_wininers[0] += winners[0]
        total_wininers[1] += winners[1]
        total_wininers[2] += winners[2]
    return win







def pit(old_model,new_model,number_of_games=25):
    # test the models against each other and accpet the new model
    # if it wins 55% of the non tie games games
    winners = {0:0,1:0,2:0}
    # winners = {'tie':0,'old_model':0,'new_model':0}
    turn = 1
    # check the two models are different
    for game in  tqdm(range(number_of_games),leave=False):
        board = np.zeros((6,7))#tictactoe_functions.get_initial_board()
        if(game < number_of_games //2):
            # old_model goes first
            player1 = old_model
            player2 = new_model

        else:
            # new model goes first
            player1 = new_model 
            player2 = old_model

        winner,experience = run_game(player1,player2,num_rollouts=20)
        # update winners array
        if(game >= number_of_games // 2):
            # flip the winner so it corresponds with the correct model
            if(winner == 2):
                winner = 1
            elif(winner == 1):
                winner = 2

        winners[winner] += 1
        # clear both players mcts tree
        player1.clear_tree()
        player2.clear_tree()

    return winners



def get_game_action_probs(mcts_model, board, turn, num_rollouts=50,temp=1.0):
	return mcts_model.get_action_probs(board,turn,num_rollouts=num_rollouts,temp=temp)


def run_game(mcts_model1,mcts_model2,temp=1.0,num_rollouts=50):

    board = np.zeros((6,7))
    experience = []
    turn = 1
    connect4_board = Connect4Logic.Board(np_pieces=board)
    for game_step in range(42):
    	# temp controls how likely 
        # the tree will explore
        # 1 for exploration
        # 0 for the pit, always take the best action
        if(turn == 1):
        	action_probs = get_game_action_probs(mcts_model1,board,turn,temp=temp,num_rollouts=num_rollouts)
        else:
        	action_probs = get_game_action_probs(mcts_model2,board,turn,temp=temp,num_rollouts=num_rollouts)


        #chose a action from these probabilities
        action = np.random.choice(7,1,p=action_probs)[0]


        if(game_step == 0):
            # truly random for first step
            action = np.random.choice(7,1)[0]


        # initially use a placeholder value for value
        # this experience is [obs, turn, action, value]
        

        # TODO:
        # should implement board flipping 
        # rotating boards to get more info
        experience.append([board,turn,action_probs,-9999])
        connect4_board.add_stone(action,turn)
        board = connect4_board.np_pieces


        winner = connect4_board.get_win_state()
        if(winner.is_ended):
        	if(winner.winner == None):
        		return 0,experience
        	return winner.winner,experience
        turn = 2 if turn == 1 else 1






def update_experience_value(winner,experience):
    # in place update experience
    winner_value = -8888
    if(winner == 1):
        # player 1 wins
        winner_value = 1 
    elif(winner == 2): 
        winner_value = -1
    elif(winner == 0):
        # tie
        winner_value = 0
    else:
        print("should get here for update experience_value")

    # doing in place to avoid gross amounts of extra memory

    # we have to ossolate winners because we flipped 
    # half the boards
    for index in range(len(experience)):
        experience[index][3] = winner_value
        # will be winner_value when turn == 1
        # and -winner_value when turn == 2
        if(index != 0 and (index) % 4 == 0):
            # because we rotated 4 times
            winner_value = -winner_value


def run_game_parllel(mcts_model1,mcts_model2,num_games,temp=1.0,num_rollouts=20):
    print("cpu_counts",cpu_count())
    pbar = tqdm(total=num_games)
    def update(*a):
        pbar.update()
    with Pool(cpu_count()) as p:

        wins = {0:0,1:0,2:0}
        loop_experience = []

        inputs = [[mcts_model1,mcts_model2,temp,num_rollouts] for x in range(num_games)]
        responses = []
        for i in range(num_games):
            res =p.apply_async(run_game_wrapper,inputs[i],callback=update)
            responses.append(res)

        # wait on all of them
        [res.wait() for res in responses]
        
        results = [x.get() for x in responses]

        #results = p.starmap(run_game_wrapper,inputs)
        for win,experience in results:
            loop_experience.extend(experience)
            wins[win] += 1

        return loop_experience,wins

def run_game_wrapper(mcts_model1,mcts_model2,temp,num_rollouts):
    winner, experience= run_game(mcts_model1,mcts_model2,temp=temp,num_rollouts=num_rollouts)
    update_experience_value(winner,experience)
    return winner,experience


def get_experience(mcts_model,temp=1.0,num_rollouts=20):
    wins = {0:0,1:0,2:0}
    loop_experience = []
    for game in tqdm(range(num_games),leave=False):
        winner, experience= run_game(mcts_model,mcts_model,temp=temp,num_rollouts=num_rollouts)
        
        # update the experience 
        # based on the real winner of the game
        update_experience_value(winner,experience)
        # check to make sure this is updated
        loop_experience.extend(experience)
        wins[winner] += 1

        # what would happen if you cleared the tree here?
        #mcts_model.clear_tree()
    return loop_experience,wins
    
Configuration = namedtuple("Configuration","columns rows inarow")





if __name__ == "__main__":
    alpha_connectx_model = None
    if(path.exists("alpha_connectx_model.torch")):
        print("Loading Prexisting model")
        alpha_connectx_model = torch.load("alpha_connectx_model.torch")

    # inialize the model
    board = np.zeros((6,7))
    config = Configuration(7,6,4)
    mcts_model = AlphaModel(config,learning_rate=1e-4,load_model=alpha_connectx_model)#tabular_mcts(policy_value_model = policy_value_model)
    mcts_model.eval()

    # num games per training loop
    num_training_loops = 500
    base_num_games = 15
    num_games = base_num_games

    temp = 1


    total_experience = []#fixed_size_list(100000)
    for train_loop in range(num_training_loops):
        loop_experience,wins = run_game_parllel(mcts_model,mcts_model,num_games,temp=temp,num_rollouts=20)
        total_experience.extend(loop_experience)

        # temp controls how close we are to using argmax
        temp *= .9
        #after we have played our games, update the model
        old_model = mcts_model.copy()
        old_model.eval()
        print("\rTraining...",end="")
        mcts_model.train_on_data(total_experience)
        total_games =  num_games
        win_averages = [x/total_games for x in wins.values()]
        print("\rTies: %.2f Player 1: %.2f Player 2: %.2f                  "%(win_averages[0],win_averages[1],win_averages[2]))


        # clear out the tree that we have built up 
        mcts_model.clear_tree()

        # run through the pit
        print("\rCompeting in the Pit...",end="")
        num_pit_games = 15
        pit_results = pit(old_model,mcts_model,number_of_games=num_pit_games)
        print("\rpit results:             ")
        print("Ties:",pit_results[0])
        print("Old Model:",pit_results[1])
        print("New Model:",pit_results[2])

        # check if the new model won more than .50 % of the no tie games
        if(pit_results[2] < (num_pit_games-pit_results[0]) * .50):
            # old_model won
            #mcts_model = tabular_mcts()
            mcts_model = old_model
            # go to before training
            num_games *= 2
            total_experience = []
            print("old model won, training for twice as long")

        else:
            print("keeping new model")
            num_games = base_num_games
            num_update_steps = 1
            # we are learning a new policy,
            # remove the old data
            total_experience = []
            print("saving model...")
            torch.save(mcts_model.model.model,"alpha_connectx_model.torch")

        if(num_games >= 64*100):
            print("thats alot of games to play, we are just going to try to play less")
            num_games = base_num_games




        print("---------")


    # save value network
   
    torch.save(mcts_model.model.model,"alpha_connectx_model.torch")
    
