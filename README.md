# Tutorial

This Project is a Tutorial to set up your model for hyperparameter search using the cluster-hyperopt repository.

In this tutorial we will create:
0. a simple model, that is not yet usable for hyperparameter search
1. a function, that will be called by the hyperparameter search algorithm
2. the config file, that defines all necessary parameters
   1. WICHTIG: Erklären wie die parameter angegeben werden!
```
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
```py