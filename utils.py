import numpy as np

def median_absolute_error(y_true, y_pred):
    return np.mean(np.median(np.abs(np.array(y_pred) - np.array(y_true)), axis=0))

def acper(y_true, y_pred, threshold=0.50):

    y_true = y_true.tolist()
    y_pred = y_pred

    for yt, yp in zip(y_true, y_pred):
        
        lower_bound = yt - (threshold * yt)
        upper_bound = yt + (threshold * yt)

        if (yp >= lower_bound) & (yp <= upper_bound):
            yield True
        else:
            yield False

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.mean(np.abs(np.array(y_pred) - np.array(y_true)) / np.array(y_true), axis=0))