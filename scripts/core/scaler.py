# NOTA:
# Este Scaler ya no se usa en el pipeline principal porque el preprocesado
# se realiza en preprocess_regression() y preprocess_golub().
# Se mantiene por si se añaden modelos que requieran escalar y.

from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit_transform(self, X, y):
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        return Xs, ys

    def inverse_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
