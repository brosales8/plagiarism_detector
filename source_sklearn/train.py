from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.neural_network import MLPClassifier

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str,default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=str, nargs='+', default= "32 64 32", metavar='Hidden',
                        help='hidden dim of model (default: 32 64 32')    
    parser.add_argument('--solver', type=str, default='lbfgs', metavar='LOSS',
                        help='Loss function (default: lbfgs)')
    parser.add_argument('--epochs', type=int, default=50, metavar='ITER',
                        help='Epochs (default: 50)')
    
    parser.add_argument('--seed', type=int, default=44, metavar='SEED',
                        help='Seed (default: 44)')
    
    
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column    
    train_y = train_data.iloc[:,0].to_numpy()
    train_x = train_data.iloc[:,1:].to_numpy()
    
    
    ## --- Your code here --- ##
    
    # Proccess hidden dimension layers to Tuple
    hidden_dim_list = args.hidden_dim[0].split()    
    hidden_dim_tuple = []
    
    # convert list of strings to list of int
    [hidden_dim_tuple.append(int(x)) for x in hidden_dim_list]        
    hidden_dim_tuple = tuple(hidden_dim_tuple)
   
    print('hidden_layer_size received: {}'.format(args.hidden_dim))    
    print('hidden_dim tuple: {}'.format(hidden_dim_tuple))

    ## TODO: Define a model 
    model = MLPClassifier(
                          hidden_layer_sizes=hidden_dim_tuple,
                          solver=args.solver,
                          learning_rate_init=args.lr,
                          max_iter=args.epochs,
                          random_state=args.seed
                         )
    
    
    ## TODO: Train the model
    model.fit(train_x, train_y)
    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
