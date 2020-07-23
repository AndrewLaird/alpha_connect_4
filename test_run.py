from kaggle_environments import evaluate, make
from alpha_model import kaggle_model
from kaggle_alpha_agent import agent



if __name__ == "__main__":
        env = make('connectx', debug=True)
        configuration = env.configuration

        model = kaggle_model(configuration)

        trainer = env.train([agent, 'Random'])
        observation = trainer.reset()
        
        model.eval()
        print(model.get_preds(observation))
