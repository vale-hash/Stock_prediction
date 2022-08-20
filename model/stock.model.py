#importing the libaries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
import requests
import json
stock_data = pd.read_csv(input('enter the csv path'))
y=stock_data.Open
x=stock_data.drop('Open', axis=1)
x.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
training_set=y_train
#feature scaling for LSTM use
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler(feature_range=(0,1))
#reshaping the values to give a 2d array before scaling
training_set_reshaped=training_set.values.reshape(-1,1)
#scaling
training_set_scaled=scale.fit_transform(training_set_reshaped)
training_set_scaled[:5]
X_train=[]
Y_train=[]
#creating a data structure with 20 intervals and 1 output
z=len(training_set)//20
for i in range(z,len(training_set)):
    X_train.append(training_set_scaled[i-40:i,0])
    Y_train.append(training_set_scaled[i,0])
X_train,Y_train=np.array(X_train),np.array(Y_train)
#reshaping
input_shape=(X_train.shape[1],1)
#Initializing the rnn model
model= LinearRegression()
model.fit(X_train,Y_train)
#predicting test data
test_set=y_test
test_set_reshaped=test_set.values.reshape(-1,1)
#scaling
test_set_scaled=scale.fit_transform(test_set_reshaped)
test_set_scaled[:5]
Total_data=pd.concat((training_set,test_set), axis = 0)
inputs=Total_data[len(Total_data)-len(test_set)-z:].values
inputs = inputs.reshape(-1,1)
inputs= scale.transform(inputs)
X_test=[]
for i in range(z,len(test_set)):
    X_test.append(test_set_scaled[i-z:i,0])
X_test=np.array(X_test)
predicted_price = model.predict(X_test)
predicted_price= predicted_price.reshape(-1,1)
predicted_price = scale.inverse_transform(predicted_price)
real_price=test_set.values.reshape(-1,1)
#displaying the result
plt.plot((predicted_price[0:len(real_price)]) , color = 'green' , label = 'prediction')
plt.title('PREDICTION ')
plt.legend()
plt.show()
