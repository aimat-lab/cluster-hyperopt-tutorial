from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Model:
    rfc = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    random_state = None

    def __init__(self, random_state=0):
        self.random_state = random_state

    def train(self, n_estimators, criterion, max_depth, bootstrap, max_features):
        self.rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          bootstrap=bootstrap, max_features=max_features, random_state=self.random_state)
        self.rfc.fit(X=self.X_train, y=self.y_train)

    def evaluate(self):
        return self.rfc.score(self.X_test, self.y_test)

    def load_dataset(self):
        #ds = datasets.load_iris()
        ds = datasets.load_wine()
        X = ds.data
        y = ds.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33,
                                                                                random_state=self.random_state)

def run():
    raise NotImplementedError

def run_hyperopt(config=None):
    suggestion = config['suggestion']
    n_estimators = suggestion['n_estimators']
    criterion = suggestion['criterion']
    max_depth = suggestion['max_depth']
    bootstrap = suggestion['bootstrap']
    max_features = suggestion['max_features']

    model = Model()
    model.load_dataset()
    model.train(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                bootstrap=bootstrap, max_features=max_features)

    result = model.evaluate()
    metadata = None
    print(result)

    return result, metadata

parameters = {'max_depth':[4,8,16,32], 'n_estimators':[4,8,16,32], 'bootstrap':[True,False], 'max_features': ['sqrt','log2',None], 'criterion':['gini','entropy']}

if __name__ == '__main__':
    config = {
        "suggestion": {
            'n_estimators': 3,
            'criterion': 'entropy',
            'max_depth': 2,
            'bootstrap': True,
            'max_features': 'sqrt'
        }
    }
    run_hyperopt(config)
