from alpha_model import kaggle_model
import numpy as np

def load_agent(tau=0.0):

    model = kaggle_model(configuration)

    def agent(observation, configuration):
        value,policy = model.get_preds(observation)
        times_visited = model.get_N(observation)
        # Return which column to drop a checker (action).
        if(tau == 0):
            action = np.argmax(times_visisted)
        else:
            weighted_times_visited = [x^(1/tau) for x in times_visited]
            weighted_sum= sum(weighted_times_visited)
            action_probs = [x/weighted_sum for x in weighted_times_visited]
            np.choice(len(weighted_times_visited),1,p=action_probs)[0]
        return action
    return agent
