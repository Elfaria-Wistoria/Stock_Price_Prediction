import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import ta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.df = None
        self.model = None
        # Menggunakan RobustScaler untuk menangani outliers
        self.scaler_price = RobustScaler()
        self.scaler_features = RobustScaler()
        
    def load_data(self, uploaded_file):
        try:
            self.df = pd.read_csv(uploaded_file)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
            
            if self.df.isnull().sum().sum() > 0:
                st.warning("Missing values detected. Filling with forward fill method.")
                self.df.fillna(method='ffill', inplace=True)
                
            # Menambahkan log returns
            self.df['Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
            return self.df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise

    def add_technical_indicators(self):
        try:
            with st.spinner("Calculating technical indicators..."):
                # Trend indicators
                self.df['MA20'] = ta.trend.sma_indicator(self.df['Close'], window=20)
                self.df['MA50'] = ta.trend.sma_indicator(self.df['Close'], window=50)
                self.df['EMA12'] = ta.trend.ema_indicator(self.df['Close'], window=12)
                self.df['EMA26'] = ta.trend.ema_indicator(self.df['Close'], window=26)
                
                # Momentum indicators
                self.df['RSI'] = ta.momentum.rsi(self.df['Close'], window=14)
                self.df['MACD'] = ta.trend.macd_diff(self.df['Close'])
                self.df['Williams_R'] = ta.momentum.williams_r(self.df['High'], self.df['Low'], self.df['Close'])
                
                # Volatility indicators
                bb_indicator = ta.volatility.BollingerBands(close=self.df['Close'])
                self.df['BB_upper'] = bb_indicator.bollinger_hband()
                self.df['BB_middle'] = bb_indicator.bollinger_mavg()
                self.df['BB_lower'] = bb_indicator.bollinger_lband()
                self.df['ATR'] = ta.volatility.AverageTrueRange(high=self.df['High'], low=self.df['Low'], close=self.df['Close']).average_true_range()
                
                # Volume indicators
                self.df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=self.df['Close'], volume=self.df['Volume']).on_balance_volume()
                
                self.df.dropna(inplace=True)
                
            return self.df
            
        except Exception as e:
            st.error(f"Error adding technical indicators: {str(e)}")
            raise

    def prepare_sequences(self, sequence_length=60):
        try:
            with st.spinner("Preparing sequences for model training..."):
                # Scale only Close price for target
                scaled_close = self.scaler_price.fit_transform(self.df[['Close']])
                
                # Select and scale features
                feature_columns = ['Close', 'Returns', 'MA20', 'MA50', 'EMA12', 'EMA26', 
                                 'RSI', 'MACD', 'Williams_R', 'BB_upper', 'BB_lower', 
                                 'ATR', 'OBV']
                scaled_features = self.scaler_features.fit_transform(self.df[feature_columns])
                
                X, y = [], []
                for i in range(len(scaled_features) - sequence_length):
                    X.append(scaled_features[i:i+sequence_length])
                    y.append(scaled_close[i+sequence_length])
                    
                X, y = np.array(X), np.array(y)
                
                # Split with larger training set
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.15, shuffle=False
                )
                
                return X_train, X_test, y_train, y_test
                
        except Exception as e:
            st.error(f"Error preparing sequences: {str(e)}")
            raise

    def build_model(self, input_shape):
        try:
            with st.spinner("Building LSTM model..."):
                self.model = Sequential([
                    # First LSTM layer with increased units
                    LSTM(256, return_sequences=True, input_shape=input_shape),
                    BatchNormalization(),
                    Dropout(0.2),
                    
                    # Second LSTM layer
                    LSTM(128, return_sequences=True),
                    BatchNormalization(),
                    Dropout(0.2),
                    
                    # Third LSTM layer
                    LSTM(64, return_sequences=True),
                    BatchNormalization(),
                    Dropout(0.2),
                    
                    # Fourth LSTM layer
                    LSTM(32, return_sequences=False),
                    BatchNormalization(),
                    Dropout(0.1),
                    
                    # Dense layers
                    Dense(64, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.1),
                    
                    Dense(32, activation='relu'),
                    BatchNormalization(),
                    
                    Dense(1, activation='linear')
                ])
                
                optimizer = Adam(learning_rate=0.0005)  # Lower learning rate
                self.model.compile(
                    optimizer=optimizer,
                    loss='huber',
                    metrics=['mae', 'mse']
                )
                
                return self.model
                
                return self.model
                
        except Exception as e:
            st.error(f"Error building model: {str(e)}")
            raise

    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        try:
            with st.spinner("Training model... This may take a while."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                class CustomCallback(EarlyStopping):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Training Progress: {int(progress * 100)}%")
                        super().on_epoch_end(epoch, logs)

                callbacks = [
                    CustomCallback(
                        monitor='val_loss',
                        patience=15,
                        restore_best_weights=True
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=0.00001
                    )
                ]
                
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0
                )
                
                status_text.text("Training completed!")
                return history
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            raise

    def plot_predictions(self, y_test, predictions):
        try:
            # Ensure correct shapes for transformation
            y_true = self.scaler_price.inverse_transform(y_test.reshape(-1, 1))
            pred_transformed = self.scaler_price.inverse_transform(predictions.reshape(-1, 1))
            
            # Create date index for x-axis
            test_dates = self.df.index[-len(y_test):]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=y_true.flatten(),
                name='Actual Price',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=pred_transformed.flatten(),
                name='Predicted Price',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title='Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600,
                template='plotly_dark',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in plotting predictions: {str(e)}")
            raise

    def evaluate_predictions(self, y_test, predictions):
        try:
            # Ensure correct shapes for transformation
            y_test_reshaped = y_test.reshape(-1, 1)
            predictions_reshaped = predictions.reshape(-1, 1)
            
            y_true = self.scaler_price.inverse_transform(y_test_reshaped)
            pred_transformed = self.scaler_price.inverse_transform(predictions_reshaped)
            
            mse = np.mean((y_true - pred_transformed) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - pred_transformed))
            mape = np.mean(np.abs((y_true - pred_transformed) / y_true)) * 100
            
            metrics = {
                'Mean Squared Error (MSE)': mse,
                'Root Mean Squared Error (RMSE)': rmse,
                'Mean Absolute Error (MAE)': mae,
                'Mean Absolute Percentage Error (MAPE)': mape
            }
            
            cols = st.columns(4)
            for i, (metric, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(metric, f"{value:.2f}")
                    
            return metrics
            
        except Exception as e:
            st.error(f"Error evaluating predictions: {str(e)}")
            raise

def main():
    st.set_page_config(page_title="Advanced Stock Price Predictor", layout="wide")
    
    st.title("📈 Advanced Stock Price Prediction App")
    st.markdown("""
    ### Enhanced LSTM Model with Advanced Technical Indicators
    This application uses a sophisticated LSTM neural network architecture with multiple technical 
    indicators to predict stock prices. The model incorporates various technical analysis features 
    and advanced preprocessing techniques for improved accuracy.
    """)
    
    uploaded_file = st.file_uploader("Upload your stock data CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            predictor = StockPredictor()
            
            with st.expander("📊 Data Preview", expanded=True):
                df = predictor.load_data(uploaded_file)
                st.dataframe(df.head())
            
            df = predictor.add_technical_indicators()
            
            with st.expander("🔧 Model Parameters", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    sequence_length = st.slider("Sequence Length", 30, 120, 60)
                with col2:
                    epochs = st.slider("Number of Epochs", 50, 200, 100)
                with col3:
                    batch_size = st.slider("Batch Size", 16, 128, 32)
            
            if st.button("🚀 Train Model"):
                X_train, X_test, y_train, y_test = predictor.prepare_sequences(sequence_length)
                model = predictor.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                history = predictor.train_model(X_train, y_train, X_test, y_test, epochs, batch_size)
                
                predictions = predictor.model.predict(X_test)
                
                st.header("📊 Results")
                predictor.plot_predictions(y_test, predictions)
                predictor.evaluate_predictions(y_test, predictions)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    st.markdown("---")
    st.markdown("Built with Streamlit and TensorFlow")

if __name__ == "__main__":
    main()