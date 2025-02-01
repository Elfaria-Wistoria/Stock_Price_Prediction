# 📈 Advanced Stock Price Predictor

## 🚀 Overview
This project is a **Stock Price Prediction App** that utilizes **Long Short-Term Memory (LSTM)** neural networks to forecast stock prices based on historical data and advanced **technical indicators**. Built using **Streamlit**, **TensorFlow**, and **Plotly**, this app provides an interactive interface for users to upload stock data, preprocess it, and train a deep learning model for price prediction.

## 🎯 Features
- **Data Upload**: Upload CSV files containing stock price data.
- **Technical Indicators**: Automatically computes key indicators such as Moving Averages, RSI, MACD, Bollinger Bands, ATR, and OBV.
- **LSTM Neural Network**: A multi-layer LSTM model optimized for time series forecasting.
- **Hyperparameter Selection**: Adjustable parameters like sequence length, epochs, and batch size.
- **Training Visualization**: Displays progress during training using Streamlit widgets.
- **Stock Price Prediction**: Predicts future stock prices and visualizes results using **Plotly**.
- **Performance Metrics**: Evaluates predictions using MSE, RMSE, MAE, and MAPE.

## 🛠️ Installation
To run this project locally, follow these steps:

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/Elfaria-Wistoria/Stock_Price_Prediction.git
cd stock-price-predictor
```

### 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application:
```bash
streamlit run app.py
```

## 📊 How It Works
1. **Upload Stock Data**: Upload a CSV file containing historical stock prices (Date, Open, High, Low, Close, Volume).
2. **Feature Engineering**: The app calculates **technical indicators** and prepares the dataset for model training.
3. **Train LSTM Model**: Users can configure training parameters and start training the deep learning model.
4. **Predict Stock Prices**: After training, the model predicts stock prices and displays results.
5. **Evaluate Model Performance**: The app calculates MSE, RMSE, MAE, and MAPE to assess prediction accuracy.

## 📌 Dependencies
- **Python 3.8+**
- **Streamlit**
- **NumPy**
- **Pandas**
- **Plotly**
- **Scikit-Learn**
- **TensorFlow**
- **TA-Lib (Technical Analysis Library)**

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

## 👨‍💻 Author
Developed by [Your Name](https://github.com/yourusername). Feel free to contribute!

---
✨ Happy Coding! 🚀

