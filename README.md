# Tutorial

This Project is a Tutorial to set up your model for hyperparameter search using the cluster-hyperopt repository.

In this tutorial we will create:

1. a simple model, that is not yet usable for hyperparameter search

2. a function, that will be called by the hyperparameter search algorithm

3. the config file, that defines all necessary parameters
   1. WICHTIG: Erklären wie die parameter angegeben werden!
   

1. We create a file ```model.py```. It contains the class ```Model```, 
which has functions to create, train and evaluate a random forest classifier imported from sklearn.
As you can see, the train function allows us to set the number of trees (```n_estimators```), 
the splitting criterion (```criterion```), the maximum depth for a tree (```max_depth```), 
if we want to use bootstrapping (```bootstrap```) and how many random features 
should be considered when calculating a split (```max_features```).
```py
from sklearn.ensemble import RandomForestClassifier


class Model:
    rfc = None      # The Random Forest Classifier

    random_state = None

    def __init__(self, random_state=0):
        self.random_state = random_state

    def train(self, X_train, y_train, n_estimators, criterion, max_depth, bootstrap, max_features):
        self.rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          bootstrap=bootstrap, max_features=max_features,
                                          random_state=self.random_state)
        self.rfc.fit(X=X_train, y=y_train)

    def evaluate(self, X_test, y_test):
        return self.rfc.score(X_test, y_test)
```

We now need a function that loads a dataset, creates a Model, trains and evaluate it. 
Therefore we create the file ```main.py``` with the function ```run_without_hyperopt()```.
This function trains the model with the following hyperparameters:
   - n_estimators = 5
   - criterion = 'entropy'
   - bootstrap = True
   - max_features = 'sqrt'

Now we can execute this function and will see the accuracy of the model in the console.
```py
from model import Model
from sklearn import datasets
from sklearn.model_selection import train_test_split

random_state = 0


def run_without_hyperopt():
    # load and split dataset
    ds = datasets.load_wine()
    X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.33, random_state=random_state)

    # train and evaluate model
    model = Model()
    model.train(X_train=X_train, y_train=y_train, n_estimators=5, criterion='entropy', max_depth=5,
                bootstrap=True, max_features='sqrt')
    print(model.evaluate(X_test=X_test, y_test=y_test))


if __name__ == '__main__':
    run_without_hyperopt()
```

2. In this second step we create a function that will be called by the 
hyperparameter optimizer and returns the accuracy and metadata of the model.
We add this function to the ```main.py``` file.
```run_hyperopt(hyperopt_config=None)``` takes a run configuration as input. 
This configuration contains the suggested hyperparameters for our model.
   1. In part 01 the function extracts the suggestion from the hyperopt_config
   and from the suggestion it can get the values for each hyperparameter
   2. Part 02 is about loading the dataset and splitting it into the train and test parts.
   3. In part 03 the function trains and evaluates the model. Afterwards it returns 
   the result (accuracy) and the metadata. 
   This is important, since the optimizer expects this return format.
```py
def run_hyperopt(hyperopt_config=None):
    # Extract options (Part 01)
    suggestion = hyperopt_config['suggestion']

    n_estimators = suggestion['n_estimators']
    criterion = suggestion['criterion']
    max_depth = suggestion['max_depth']
    bootstrap = suggestion['bootstrap']
    max_features = suggestion['max_features']

    # load and split dataset (Part 02)
    # this is not optimal, since we only need to download the dataset once. We only use this for simplicity
    ds = datasets.load_wine()
    X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.33, random_state=random_state)

    # train and evaluate model (Part 03)
    model = Model()
    model.train(X_train=X_train, y_train=y_train, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                bootstrap=bootstrap, max_features=max_features)

    result = model.evaluate(X_test=X_test, y_test=y_test)
    metadata = None
    print(result)

    return result, metadata
```
Right now it is not yet possible to start the hyperparameter search. We first need to write a 
configuration file, which defines all parameters needed for the hyperparameter search.

3. In the 3rd Section we create the configuration file needed to run the hyperparameter search. 
Do not confuse this configuration file with the hyperopt_config parameter
for the ```run_hyperopt(hyperopt_config=None)``` function, created in section 2.
