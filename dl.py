import pandas as pd
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/dl/best_model.h5")
def predict(lst):
    # test: '''np.array([[3.0,1.0,34.5,7.8292,0.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0]'''
    X_test = np.array([lst], dtype='float32')
    res = {0:'Không sống sót', 1:'Sống sót'}
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return res[y_pred[0][0]]