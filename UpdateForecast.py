import requests
import json
import numpy as np
import pandas as pd
from datetime import date,datetime,timezone
from forecast import predict

def update():
    currrent_daytime= datetime.now()
    dt = int((currrent_daytime - datetime(1970, 1, 1)).total_seconds())
    dt-=86400
    m=str(dt)
    url='https://api.openweathermap.org/data/2.5/onecall/timemachine?lat=38.7996&lon=26.9707&units=metric&dt='+m+'&appid=965365707dabce3b9549ba434d715f3b'
    response = requests.get(url)
    data_dict = response.text
    parsed = json.loads(data_dict)
    data = pd.DataFrame.from_dict(parsed['hourly'])

    wind_data=data.drop(['dew_point','feels_like', 'clouds', 'weather','visibility','wind_deg'], axis=1)

    for i in range(len(wind_data['dt'])):
        wind_data['dt'].iloc[i]=datetime.utcfromtimestamp(wind_data['dt'].iloc[i]).strftime('%Y-%m-%d %H:%M:%S')
    wind_data['dt']=pd.to_datetime(wind_data['dt'])
    wind_data=wind_data[['dt','temp','pressure','humidity','wind_speed']]

    train_data=pd.read_csv('wind_data.csv')
    train_data=train_data.append(wind_data, ignore_index=True)
    train_data.to_csv('wind_data.csv', index=False)

    predict()
    return