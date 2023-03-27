import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU



def randomForest():
    # Import dataset
    #The code reads a csv file named "RELIANCE.csv" using pandas and stores it in a dataframe called "bist100".
    bist100 = pd.read_csv("RELIANCE.csv")
    # Rename columns
    bist100.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"},
                   inplace=True)
    # Checking null value
    bist100.isnull().sum()
    # Checking na value
    bist100.isna().any()
    # The code drops any rows with null values using the "dropna()" method .
    bist100.dropna(inplace=True)
    #The code is used to check if there are any na values in the dataframe after dropping the null values.
    bist100.isna().any()
    # convert date field from string to Date format and make it index
    bist100['date'] = pd.to_datetime(bist100.date)
    # sorting dataset by date format
    bist100.sort_values(by='date', inplace=True)
    # Get the duration of dataset
    # The time of the first bar of data
    print("Starting date: ", bist100.iloc[0][0])
    # Time of the last piece of data
    print("Ending date: ", bist100.iloc[-1][0])
    #duration
    print("Duration: ", bist100.iloc[-1][0] - bist100.iloc[0][0])
    # Monthwise High and Low stock price
    bist100.groupby(bist100['date'].dt.strftime('%B'))['low'].min()
    #Keep close date data
    closedf = bist100[['date', 'close']]
    #Make a copy of the data for easy use
    close_stock = closedf.copy()
    #Delete date, leaving only close
    del closedf['date']
    # Maximum minimization normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf))
    # Training data 0.65 Test data 0.35
    training_size = int(len(closedf) * 0.65)
    test_size = len(closedf) - training_size
    # Divide the data set according to the index divided above
    train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :]
    # Divide the data set according to the time window
    # Using two weeks' worth of data to predict one day's worth of data
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3------15
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    # Using two weeks' worth of data to predict one day's worth of data
    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    from sklearn.ensemble import RandomForestRegressor
    # Build a random forest model
    regressor = RandomForestRegressor(n_estimators=5)
    # Training model
    regressor.fit(X_train, y_train)
    # Lets Do the prediction
    train_predict = regressor.predict(X_train)
    test_predict = regressor.predict(X_test)
    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)
    # From maximum to minimum normalization to its original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))
    # Evaluation metrices MSE
    print("Random forest MSE: ", mean_squared_error(original_ytest, test_predict))

def lstm_one_week():
    # Import dataset
    # The code reads a csv file named "RELIANCE.csv" using pandas and stores it in a dataframe called "bist100".
    bist100 = pd.read_csv("RELIANCE.csv")
    # Rename columns
    bist100.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"},
                   inplace=True)
    # Checking null value
    bist100.isnull().sum()
    # Checking na value
    bist100.isna().any()
    # The code drops any rows with null values using the "dropna()" method .
    bist100.dropna(inplace=True)
    # The code is used to check if there are any na values in the dataframe after dropping the null values.
    bist100.isna().any()
    # convert date field from string to Date format and make it index
    bist100['date'] = pd.to_datetime(bist100.date)
    # sorting dataset by date format
    bist100.sort_values(by='date', inplace=True)
    # Get the duration of dataset
    # The time of the first bar of data
    print("Starting date: ", bist100.iloc[0][0])
    # Time of the last piece of data
    print("Ending date: ", bist100.iloc[-1][0])
    # duration
    print("Duration: ", bist100.iloc[-1][0] - bist100.iloc[0][0])
    # Monthwise High and Low stock price
    bist100.groupby(bist100['date'].dt.strftime('%B'))['low'].min()
    # Keep close date data
    closedf = bist100[['date', 'close']]
    # Make a copy of the data for easy use
    close_stock = closedf.copy()

    # Delete date, leaving only close
    del closedf['date']

    # Maximum minimization normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf))

    # Training data 0.65 Test data 0.35
    training_size = int(len(closedf) * 0.65)
    test_size = len(closedf) - training_size

    # Divide the data set according to the index divided above
    train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :]

    # Divide the data set according to the time window
    # Using one weeks' worth of data to predict one day's worth of data
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        #For each prediction, data from the previous window is used
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3,4,
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])#X(0,1,2,3,4)->Y(5)  X(1,2,3,4,5)->Y(6)
        return np.array(dataX), np.array(dataY)

    #  Using one weeks' worth of data to predict one day's worth of data
    time_step = 5 #5 is one week, 10 is two weeks
    #Get the training set and test machine with input lstm
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
#*********************************************************same***************************************************
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    #Clearing a session
    tf.keras.backend.clear_session()

    #Create the lstm model
    model = Sequential()
    #add input layer
    model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
    #add lstm layer
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    #add dense layer
    model.add(Dense(1))

    #Defined loss function as MSE
    model.compile(loss='mean_squared_error', optimizer='adam')

    #Training model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=5, verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    #Converts the max-min normalization to the initial value
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation metrices MSE
    print("one-week-lstm MSE: ", mean_squared_error(original_ytest, test_predict))


def lstm_two_week():
    # Import dataset
    # The code reads a csv file named "RELIANCE.csv" using pandas and stores it in a dataframe called "bist100".
    bist100 = pd.read_csv("RELIANCE.csv")
    # Rename columns
    bist100.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"},
                   inplace=True)
    # Checking null value
    bist100.isnull().sum()
    # Checking na value
    bist100.isna().any()
    # The code drops any rows with null values using the "dropna()" method .
    bist100.dropna(inplace=True)
    # The code is used to check if there are any na values in the dataframe after dropping the null values.
    bist100.isna().any()
    # convert date field from string to Date format and make it index
    bist100['date'] = pd.to_datetime(bist100.date)
    # sorting dataset by date format
    bist100.sort_values(by='date', inplace=True)
    # Get the duration of dataset
    # The time of the first bar of data
    print("Starting date: ", bist100.iloc[0][0])
    # Time of the last piece of data
    print("Ending date: ", bist100.iloc[-1][0])
    # duration
    print("Duration: ", bist100.iloc[-1][0] - bist100.iloc[0][0])
    # Monthwise High and Low stock price
    bist100.groupby(bist100['date'].dt.strftime('%B'))['low'].min()
    # Keep close date data
    closedf = bist100[['date', 'close']]
    # Make a copy of the data for easy use
    close_stock = closedf.copy()

    # Delete date, leaving only close
    del closedf['date']

    # Maximum minimization normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf))

    # Training data 0.65 Test data 0.35
    training_size = int(len(closedf) * 0.65)
    test_size = len(closedf) - training_size

    # Divide the data set according to the index divided above
    train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :]

    # Divide the data set according to the time window
    # Using two weeks' worth of data to predict one day's worth of data
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3,4,5,,,,9
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])#X(0,1,2,3,4,...9)->Y(10)
        return np.array(dataX), np.array(dataY)

    #  Using two weeks' worth of data to predict one day's worth of data
    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
#*********************************************************same***************************************************
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Clearing a session
    tf.keras.backend.clear_session()

    # Create the lstm model
    model = Sequential()
    # add input layer
    model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
    # add lstm layer
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    # add dense layer
    model.add(Dense(1))

    # Defined loss function as MSE
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Training model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=5, verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Converts the max-min normalization to the initial value
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation metrices MSE
    print("two-week-lstm MSE: ", mean_squared_error(original_ytest, test_predict))


#Experiment 1 Evaluation of machine learning performance
#randomForest()
#lstm with a two-week window
lstm_two_week()
#lstm with a one-week window
#lstm_one_week()
