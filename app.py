import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def main():
    st.title("Stock Price Predictor App")

    stock = st.text_input("Enter the Stock ID", "GOOG")
    end = datetime.now()
    start = datetime(end.year-10, end.month, end.day)

    data = yf.download(stock, end=end, start=start)

    st.subheader("Stock Data")
    st.write(data.describe())


    st.subheader("Closing Price vs Time chart")
    data['Close'].plot(figsize=(16,6))
    fig1 = plt.title('Close Price History')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid()
    st.pyplot(fig1)

    st.subheader("Closing Price vs Time chart with 100MA & 200MA")
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(data.Close)
    plt.plot(ma100)
    plt.plot(ma200)
    st.pyplot(fig2)

    scaler = MinMaxScaler(feature_range=(0, 1))
    Data = scaler.fit_transform(Data)

    X_train, X_test, X_val, y_train, y_test, y_val = create_data(Data, n_future=1, n_past=60, train_test_split_percentage=0.8,
                                               validation_split_percentage = 0)
    model = load_model("latest_model.keras")

    # Making prediction

    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    st.subheader('Prediction vs Real')
    fig3 = plot_prediction(y_test, y_pred, stock)
    st.pyplot(fig3)



def create_data(df, n_future, n_past, train_test_split_percentage, validation_split_percentage):
    n_feature = df.shape[1]
    x_data, y_data = [], []
    for i in range(n_past, len(df) - n_future + 1):
        x_data.append(df[i - n_past:i, 0:n_feature])
        y_data.append(df[i + n_future - 1:i + n_future, 0])
    
    split_training_test_starting_point = int(round(train_test_split_percentage*len(x_data)))
    split_train_validation_starting_point = int(round(split_training_test_starting_point*(1-validation_split_percentage)))
    
    x_train = x_data[:split_train_validation_starting_point]
    y_train = y_data[:split_train_validation_starting_point]
    
    # if you want to choose the validation set by yourself, uncomment the below code.
    x_val = x_data[split_train_validation_starting_point:split_training_test_starting_point]
    y_val =  x_data[split_train_validation_starting_point:split_training_test_starting_point]                                             
    
    x_test = x_data[split_training_test_starting_point:]
    y_test = y_data[split_training_test_starting_point:]
    
    return np.array(x_train), np.array(x_test), np.array(x_val), np.array(y_train), np.array(y_test), np.array(y_val)

def plot_prediction(test,prediction, stock):
    plt.plot(test,color='red',label="Real")
    plt.plot(prediction, color="blue",label="Predicted")
    plt.title(f"{stock} Prediction")
    plt.xlabel("Date")
    plt.ylabel(f"{stock}")
    plt.legend()
    plt.show()

main()