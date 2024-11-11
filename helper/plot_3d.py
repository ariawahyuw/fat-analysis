import numpy as np

def make_meshgrid(x, y, h=.05):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, X_closest, xx, yy, type=None, **params):
    Z = clf.predict(X_closest)
    if type == "multi":
        Z = np.clip(Z, 0, 1)
        Z = np.argmax(Z, axis=1)
    elif type == "binary":
        Z = np.where(Z[:, 4] > 0.0, 1, 0)
        Z = np.clip(Z, 0, 1)
    else:
        Z = Z
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out