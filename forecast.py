from utils import cycle_encode, retrive_prediction
import pandas as pd
import json
from statsmodels.tsa.ar_model import AutoRegResults
import numpy as np

def predict():
	data=pd.read_csv('wind_data.csv')

	model = AutoRegResults.load('var_model.pkl')

	data['dt']=pd.to_datetime(data['dt'])
	data.set_index('dt', inplace=True)

	data['month'] = data.index.month
	data['hour'] = data.index.hour


	data = cycle_encode(data, ['month','hour'])
	data.drop(['month','hour'], axis=1, inplace=True)

	period = 24
	data_diff = data.diff(period).dropna()
	final_pred = retrive_prediction(model, data_diff, data, 72)

	wind_speed=np.power(final_pred[:,3],3)
	pressure=final_pred[:,1]
	temp=final_pred[:,0]+273

	R=2.90515100
	air_density=np.multiply(temp,R).astype(float)
	air_density=np.divide(pressure,air_density).astype(float)

	A=3.14*58.5*58.5
	constant=A*0.5*1.173*0.5
	power_op=np.multiply(wind_speed,constant)

	power_data=pd.DataFrame(list(zip(power_op,final_pred[:,3])), columns=['Calc_power','Wind_speed'])
	power_dict=power_data.to_dict()
	power_dict = {'Points': power_dict}
	with open("db.json", "w") as wf:
		json.dump(power_dict,wf)
	return