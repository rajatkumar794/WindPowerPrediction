import numpy as np
import pandas as pd

def cycle_encode(dat, cols):

    for col in cols:
        
        dat[col + '_sin'] = np.sin(2 * np.pi * dat[col]/dat[col].max())
        dat[col + '_cos'] = np.cos(2 * np.pi * dat[col]/dat[col].max())
        
    return dat


def retrive_prediction(var_result, prior, prior_init, steps):
    period=24
    pred = var_result.forecast(np.asarray(prior), steps=steps)
    init = prior_init.tail(period).values
    
    if steps >period:
        id_period = list(range(period))*(steps//period)
        id_period = id_period + list(range(steps-len(id_period)))
    else:
        id_period = list(range(steps))
    
    final_pred = np.zeros((steps, prior.shape[1]))
    for j, (i,p) in enumerate(zip(id_period, pred)):
        final_pred[j] = init[i]+p
        init[i] = init[i]+p 
        
    return final_pred


