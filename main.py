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


def run_hyperopt(config=None):
    # Extract options
    suggestion = config['suggestion']

    n_estimators = suggestion['n_estimators']
    criterion = suggestion['criterion']
    max_depth = suggestion['max_depth']
    bootstrap = suggestion['bootstrap']
    max_features = suggestion['max_features']
    
    print("config_dict:", config)

    # load and split dataset
    # this is not optimal, since we only need to download the dataset once. We only use this for simplicity
    ds = datasets.load_wine()
    X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.33, random_state=random_state)

    # train and evaluate model
    model = Model()
    model.train(X_train=X_train, y_train=y_train, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                bootstrap=bootstrap, max_features=max_features)

    result = model.evaluate(X_test=X_test, y_test=y_test)
    metadata = None
    print(result)

    return result, metadata


if __name__ == '__main__':
    run_without_hyperopt()
