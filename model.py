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
