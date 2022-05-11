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