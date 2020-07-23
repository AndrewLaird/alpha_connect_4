# create a general model that takes in the board, does a number
# of rollouts, (depending on time remaining?) and produces an answer as to which action its taking

# also responsible for training 


# AlphaGoZero used 40 residual layers 
# then on value, one conv filter, and 2 FC layers
# on policy, 2 wide? conv filter and 1 FC layer
# all of these with BatchNorm and ReLU after each layer

# residual layer is just convolutions with a skip connection over them
# this keeps the initial input throughout the network but allows work to be done on the input


# Look at the interesting gamestate in alphazero
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.modules.activation import ReLU, Tanh
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss
import numpy as np
from collections import namedtuple
import Connect4Logic
import math

def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

def convPool(in_channels,out_channels,board_size=(7,6)):
    # takes a 7x6 down to a 1x1
    return nn.Conv2d(in_channels,out_channels,kernel_size=board_size,stride=1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv(nf,nf)
        self.batch1 = nn.BatchNorm2d(nf)
        self.conv2 = conv(nf,nf)
        self.batch2 = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU()

    def forward(self, x):
        input_copy = x
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        x = self.batch1(x)
        return input_copy+x


class ResNetModel(nn.Module):
    def __init__(self,board_width,board_height):
        super(ResNetModel,self).__init__()
        # thie size of the input board
        input_size = (board_width,board_height)
        # how many options there are to chose from 
        output_size = board_width

        # 8 resblocks
        # then pool down to 1x1x16
        # then flatten for hidden layers
        self.hidden_model = nn.Sequential(
                conv(5,16),
                ResBlock(16),
                ResBlock(16),
                ResBlock(16),
                ResBlock(16),
                #ResBlock(16),
                #ResBlock(16),
                #ResBlock(16),
                #ResBlock(16),
                convPool(16,16,board_size=input_size),
                Flatten(),
        )
        self.value_model = nn.Sequential(
                nn.Linear(16,16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16,1)
        )
        self.policy_model = nn.Sequential(
                nn.Linear(16,16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16,output_size)
        )


    def forward(self,x):
        hidden_rep = self.hidden_model(x)

        value = self.value_model(hidden_rep)
        policy = self.policy_model(hidden_rep)

        return policy,value


class model_wrapper():
    # has all the preprocessing and training plug ins
    # while alphamodel is just the pytorch nnet
    def __init__(self,configuration,learning_rate=1e-3,load_model=None):
        # load any saved models
        # Number of Columns on the Board.
        self.columns = configuration.columns
        # Number of Rows on the Board.
        self.rows = configuration.rows
        # Number of Checkers "in a row" needed to win.
        self.inarow = configuration.inarow

        if(load_model == None):
            self.model = ResNetModel(self.columns,self.rows) 
        else:
            self.model = load_model

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.policy_loss = nn.KLDivLoss()
        #self.policy_loss = nn.PoissonNLLLoss()
        # self.policy_loss = MSELoss()
        # self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = MSELoss()

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def preprocess(self,board,turn):
        # take something that is of the form 
        # [0,1,2,2,1,0..]
        #     .
        #     .
        #(cols x rows)

        # transform it into a 3 dimensional input for our neural network like
        # the alpha zero paper
        # one layer for who's turn it is
        # one layer for how many you have to connect 
        # one layer for player 1's pieces
        # one layer for player 2's pieces
        # one layer that is the original observation

        # The current serialized Board (rows x columns).
        # how do we do this at scale?
        torch_board = torch.Tensor(board).reshape(self.columns,self.rows)
        # Which player the agent is playing as (1 or 2).
        mark = turn
        our_piece = mark
        opponent_piece = (2 if our_piece == 1 else 1)
        whos_turn = torch.ones(self.columns,self.rows) * int(mark)
        connect_x = torch.ones(self.columns,self.rows) * self.inarow

        our_pieces = (torch_board == our_piece).float()
        opponent_pieces = (torch_board == opponent_piece).float()
        

        full_observation = torch.stack([whos_turn,connect_x,our_pieces,opponent_pieces,torch_board]).unsqueeze(0)

        return full_observation

    def forward(self,board,turn):
        if(type(turn) != tuple):
            board = torch.Tensor(board).unsqueeze(0)
            turn = [turn]

        full_observation = []
        for i in range(len(turn)):
            full_observation.append(self.preprocess(board[i],turn[i]))
        #full_observation = self.preprocess(board,turn)
        full_observation = torch.cat(full_observation)
        policy,value = self.model(full_observation)

        return policy,value

    def __call__(self,board,turn):
        return self.forward(board,turn)

    def train_on_data(self, data):
        # set into training mode
        self.model.train()
        # shuffle data before we do anything with it
        np.random.shuffle(data)



        # data should be in state,action,new_state,reward
        obs,turns,actions,values = zip(*data)
        # reformat data 
        obs = torch.Tensor(obs).float()
        values = torch.Tensor(values).float().view((-1,1))

        # feed these through the network
        # cross entropy loss requires an index so we are taking the best
        # action as our index
        #actions = torch.Tensor(np.argmax(actions,axis=1)).long()
        actions = torch.Tensor(actions).float()


        epochs = 25
        for i in range(epochs):
            # Update the value network
            self.optimizer.zero_grad()

            policy_prediction, value_prediction = self.forward(obs,turns)
            v_loss = self.value_loss(value_prediction,values)

            # compute policy loss
            p_loss = self.policy_loss(policy_prediction,actions)

            total_loss = v_loss+p_loss
            total_loss.backward()
            print("\rLoss:"+str(total_loss)+"                  ",end="")

            self.optimizer.step()

        # reset into eval mode
        self.model.eval()



class AlphaModel():
    # keeps track of the tree for mcts
    # seperate from the model
    def __init__(self,configuration,learning_rate=1e-3,load_model=None):
        self.configuration = configuration
        self.Q = {}
        self.N = {}
        self.P = {}
        self.learning_rate =learning_rate
        self.number_actions = configuration.columns
        self.model = model_wrapper(configuration,load_model=load_model,learning_rate=learning_rate)
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def train_on_data(self,data):
        self.model.train_on_data(data)



    def clear_tree(self):
        # clears everything except what the model has learned
        self.Q = {}
        self.N = {}
        self.P = {}
    
    def serialize_board(self,board):
        # we know numbers will only be 0,1,2
        # so we are going to join them into a string
        serialized_string = "".join([str(x) for x in board])
        return serialized_string

    def get_Q(self,serialized_board):
        return self.Q[serialized_board]

    def get_N(self, serialized_board):
        return self.N[serialized_board]

    def get_P(self, serialized_board):
        return self.P[serialized_board]

    def update_Q(self,serialized_board,action,value):
        # update the moving average for all rotated boards
        N_state_action = self.get_N(serialized_board)[action]
        Q_state_action = self.get_Q(serialized_board)[action]
        # update the moving average

        self.Q[serialized_board][action]  = (N_state_action*Q_state_action + value) / (N_state_action + 1)

    def increment_N(self,serialized_board,action):
        self.N[serialized_board][action]  += 1

    def set_P(self,serialized_board,policy):
        self.P[serialized_board] = policy

    def get_action_probs(self,board,turn,num_rollouts=50,temp=1.0):
        serialized_board = self.serialize_board(board)
        logic_board = Connect4Logic.Board(np_pieces=board)
        for i in range(num_rollouts):
            self.rollout(board,logic_board,turn,debug=False)


        # modified from: https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py

        counts = self.N[serialized_board]

        #apply a mask to the counts to only look at valid moves
        counts = counts * logic_board.get_valid_moves()

        if(temp <= 0.01):
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            action_probs = [0] * len(counts)
            action_probs[bestA] = 1
            return action_probs

        # return weighted average
        weighted_counts = [x**(1/temp) for x in counts]
        sum_counts = float(sum(weighted_counts))
        action_probs = [x/sum_counts for x in weighted_counts]

        return action_probs

        
    def expand_node(self,board,turn):
        # first add this to Q,N
        serialized_board =  self.serialize_board(board)

        self.Q[serialized_board] = [0 for i in range(self.number_actions)]
        self.N[serialized_board] = [0 for i in range(self.number_actions)]

        predicted_policy, predicted_value = self.model(board,turn)
        predicted_policy = predicted_policy[0]
        predicted_value = predicted_value[0]

        predicted_policy = [float(x) for x in predicted_policy]
        self.set_P(serialized_board, predicted_policy)
        predicted_value = float(predicted_value)
        return predicted_value

    def seen(self,serialized_board):
        return serialized_board in self.N.keys()

    def check_for_winner(self,logic_board):
        # turn into connect4 object
        winner = logic_board.get_win_state()
        if(winner.is_ended):
            #someone has won
            if(winner.winner == None):
                # return 0 instead of none on a tie
                return 0#winner.winner = 0
            return winner.winner
        else:
            return -1

    def get_mcts_action(self,serialized_board,logic_board,c_puct=1.0):
        # to know which node we want to select to explore we use an interesting formula
        # which is U[s][a] = Q[s][a] + c_puct*P[s][a] * sqrt(sum(N[s]))/(1+N[s][a])

        # U stands for Upper confidence bound

        best_U =  -float("inf")
        multiple_best_U = []
        best_action = 0

        U_list = []
        # 
        N_s = self.get_N(serialized_board)
        Q_s = self.get_Q(serialized_board)
        P_s = self.get_P(serialized_board)
        # only look at valid action
        valid_mask = logic_board.get_valid_moves()
        valid_actions = [x for x in range(self.number_actions) if valid_mask[x]]
        for action in valid_actions:#range(self.number_actions):

            Q_s_a = Q_s[action]
            P_s_a = P_s[action]

            # N_term = math.sqrt(sum(N_s))/(N_s[action]+1)
            N_term = math.sqrt(sum(N_s))/(N_s[action]+1)

            #U = Q_s_a + c_puct*P_s_a*N_term
            U = Q_s_a + c_puct*N_term


            U_list.append((action,U))

            if(U > best_U):
                best_U = U
                best_action = action
        return U_list


    def rollout(self,board,logic_board,turn,debug=False):
        # we have the valid actions
        # and their outcomes
        serialized_board = self.serialize_board(board)

        # check if this is a winning board
        winner = self.check_for_winner(logic_board)
        if(winner != -1):
            return -winner

        # check board to see if we have seen this before
        if(not self.seen(serialized_board)):
                
            # we haven't seen this board before and we know
            # its not a winning board
            # gotta expand this node
            predicted_value = self.expand_node(board,turn)
            return  -predicted_value

        # we have seen this position before, so we have to go deeper
        U_list = self.get_mcts_action(serialized_board,logic_board,c_puct=1.0)
        # get the action with the largest U and chose randomly between tied ones
        best_actions = []
        best_action_value = float('-inf')
        for action,U in U_list:
            if(U > best_action_value):
                best_action_value = U
                best_actions = [action]
            elif(U == best_action_value):
                best_actions.append(action)
        # chose randomly between these acionts
        best_action = np.random.choice(best_actions)



        if(debug):
            Q_s = self.get_Q(serialized_board)
            P_s = self.get_P(serialized_board)
            print("--------------")
            print("board:")
            print(serialized_board)
            print("U:",U_list)
            print("N:",self.N[serialized_board])
            print("Q:",Q_s)
            print("P:",P_s)
            print("best one", best_action)


            
        # print(best_action
        logic_board = Connect4Logic.Board(np_pieces=board.copy())
        logic_board.add_stone(best_action,turn)
        board_after_action = logic_board.np_pieces

        next_turn = 2 if turn == 1 else 1
        # print("\tgoing down on this board",board,action)
        value_below = self.rollout(board_after_action,logic_board,next_turn,debug=debug)
        # update Q and N
        # updating these values on the flipped board
        self.update_Q(serialized_board, best_action, value_below)
        self.increment_N(serialized_board, best_action)

        return -value_below

    def copy(self):
        # copy both neural networks
        model_copy = ResNetModel(self.configuration.columns,self.configuration.rows)#torch_policy_value_model()
        model_copy.load_state_dict(self.model.model.state_dict())
        copy = AlphaModel(self.configuration,load_model=model_copy,learning_rate=self.learning_rate)
        return copy


Configuration = namedtuple("Configuration","columns rows inarow")


if __name__ == "__main__":
    board = np.zeros((6,7))
    config = Configuration(7,6,4)
    model = AlphaModel(config)
    model.eval()
    action_probs = model.get_action_probs(board,1,num_rollouts=1000)
    print(action_probs)




