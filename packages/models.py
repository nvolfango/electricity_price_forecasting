import pandas as pd
import datetime as dt
import numpy as np
import cloudpickle
import matplotlib as mpl
from functools import reduce
import time
import warnings
import joblib as jl

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import initializers

import statsmodels.tsa as tsa
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import ensemble
from sklearn import linear_model
import sklearn.preprocessing as prep

# Configure GPU (if available)
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except Exception as e:
    print(e)

###########################################################################################################################################
# Naive (Benchmarks)

class naive():
    """
    Naive model implementation.
    
    Parameters
    ----------
        lag : int
            Number of days to go back in order to make forecast, e.g. day_lag=7 indicates
            a weekly persistent model, day_lag=1 indicates a daily persistent model, etc.
        period : str in ['D', 'Y']
    """
    def __init__(self, lag, period):
        self.lag = lag
        self.period = period
    
    def ingest_data(self, train_target, train_bm, train_planned, train_bid_curves):
        self.data = train_target.copy()
       
    def train(self):
        # Date for which forecast will be made
        self.target_date = dt.datetime.combine(self.data.index.date[-1], dt.datetime.min.time()) + dt.timedelta(days=1)
    
    def forecast(self):
        # Make forecasts
        if self.period == 'D':
            forecast_df = self.data.loc[self.data.index.date == self.data.index.date[-(self.lag*24)]]
        elif self.period == 'Y':
            # Get corresponding date from lag years before    
            try:
                forecast_date = dt.datetime(self.target_date.year-1, self.target_date.month, self.target_date.day)
            except:
                forecast_date = dt.datetime(self.target_date.year-1, self.target_date.month, self.target_date.day-1)
            
            forecast_df = self.data.loc[self.data.index.date == forecast_date.date()]

        # Reindex forecasts to the appropriate forecast date and relabel forecasts_df
        forecast_df.index = pd.date_range(self.target_date, periods=24, freq='H')
        forecast_df.index.name = 'DeliveryPeriod'

        return(forecast_df)

############################################################################################################################################
# Random Forest

class random_forest():
    """
    Random forest model implementation.
    
    Parameters
    ----------
        model_params : dict
            n_estimators : int
                Number of trees
            max_depth : int
                Max tree depth
            max_features : int
                Number of features considered at each decision tree split
            n_jobs : int
                Number of processes (joblib argument for parallelisation)
        lag_params : dict
            price_lags : list of int
                Lags of day-ahead market price data to use as predictors
            bm_price_lags : list of int
                Lags of balancing market price data to use as predictors
            planned_lags : list of int
                Lags of forecast and wind data to use as predictors
    """
    def __init__(self, model_params, lag_params):
        self.model = ensemble.RandomForestRegressor(**model_params, oob_score=True, random_state=1)
        self.price_lags = lag_params['price_lags']
        self.bm_price_lags = lag_params['bm_price_lags']
        self.planned_lags = lag_params['planned_lags']
        
    def ingest_data(self, train_target, train_bm, train_planned, train_bid_curves):
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)

        # Split into training predictors and test predictors
        last_day_of_data = dt.datetime.combine(train_planned.index.date[-1], dt.datetime.min.time())
        test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq='H')
    
        # Initialise predictors dataframe
        predictors = pd.DataFrame(index=train_planned.index)
        
        # Build predictors from EURPrices
        for lag in self.price_lags:
            predictor_name = f"{train_target.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_target)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)
        
        # Build predictors from ImbalancePrice
        for lag in self.bm_price_lags:
            predictor_name = f"{train_bm.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_bm)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Build predictors from Wind and Demand
        for column in train_planned:
            for lag in self.planned_lags:
                predictor_name = f"{column}-{lag}"
                predictors.insert(predictors.shape[1], predictor_name, train_planned[column])
                predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Split predictors into training and test predictors and store data for training and forecasting
        train_predictors = predictors.drop(test_index)
        test_predictors = predictors.loc[test_index,:]
        
        # Remove rows in training set with NAs
        notna_train_predictors_loc = train_predictors.notna().all(axis=1)
        train_predictors = train_predictors.loc[notna_train_predictors_loc]
        train_target = train_target.loc[notna_train_predictors_loc]
        
        # Store data into object
        self.train_predictors = train_predictors
        self.test_predictors = test_predictors
        self.train_target = train_target
        
        # Initialise dataframe for variable importances
        if not hasattr(self, 'variable_importances'):
            self.variable_importances = pd.DataFrame(index=train_predictors.columns)
        
    def train(self):
        # Fit the forecasting model
        self.model.fit(X=self.train_predictors, y=self.train_target['EURPrices'])
        
        # Store variable importances
        variable_importances = self.model.feature_importances_
        self.variable_importances.insert(self.variable_importances.shape[1], self.test_predictors.index.date[0], variable_importances)

    def forecast(self):
        forecast_df = pd.DataFrame(self.model.predict(self.test_predictors), index=self.test_predictors.index)
        forecast_df.index.name = 'DeliveryPeriod'
        return(forecast_df)

############################################################################################################################################
# AR/ARX

"""
Parameters:
    model_params: dict with keys: lags, trend, ic, exog
    lag_params: dict with keys: bm_price_lags, planned_lags
"""
class ARX():
    """
    AR/ARX model implementation.
    
    Parameters
    ----------
        model_params : dict
            lags : list of int
                AR/ARX model orders to fit.
            trend : str in ['n', 'c', 't', 'ct']
                Specifies ar_model.AutoReg() parameter for model trend, if any.
            ic : str in ['aic', 'bic']
                Information criterion to use in determining the 'best' model order specification.
            exog : bool
                Whether or not to include exogenous data (i.e. implement ARX (True) or AR (False))
        lag_params : dict
            bm_price_lags : list of int
                Lags of balancing market price data to use as predictors
            planned_lags : list of int
                Lags of forecast and wind data to use as predictors
    """
    def __init__(self, model_params, lag_params):
        self.lags = model_params['lags']
        self.trend = model_params['trend']
        self.ic = model_params['ic']
        self.exog = model_params['exog']
        
        self.bm_price_lags = lag_params['bm_price_lags']
        self.planned_lags = lag_params['planned_lags']
        
    def ingest_data(self, train_target, train_bm, train_planned, train_bid_curves):
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)

        # Split into training predictors and test predictors
        last_day_of_data = dt.datetime.combine(train_planned.index.date[-1], dt.datetime.min.time())
        test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq='H')
    
        # Initialise predictors dataframe
        predictors = pd.DataFrame(index=train_planned.index)
        
        # Build predictors from ImbalancePrice
        for lag in self.bm_price_lags:
            predictor_name = f"{train_bm.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_bm)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Build predictors from Wind and Demand
        for column in train_planned:
            for lag in self.planned_lags:
                predictor_name = f"{column}-{lag}"
                predictors.insert(predictors.shape[1], predictor_name, train_planned[column])
                predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Split predictors into training and test predictors and store data for training and forecasting
        train_predictors = predictors.drop(test_index)
        test_predictors = predictors.loc[test_index,:]
        
        # Remove rows in training set with NAs
        notna_train_predictors_loc = train_predictors.notna().all(axis=1)
        train_predictors = train_predictors.loc[notna_train_predictors_loc]
        train_target = train_target.loc[notna_train_predictors_loc]
        
        # Store data into object
        self.train_predictors = train_predictors
        self.test_predictors = test_predictors
        self.train_target = train_target
        
        # Initialise dataframe for variable importances
#         if not hasattr(self, "variable_importances"):
#             self.variable_importances = pd.DataFrame(index=train_predictors.columns)
        
    def train(self):
        # Fit AR/ARX models with different lag values
        if self.exog:
            model_selection = jl.Parallel(n_jobs=-1, backend='threading') \
                (jl.delayed(tsa.ar_model.AutoReg(self.train_target.values,lag,self.trend,exog=self.train_predictors.values).fit)() for lag in self.lags)
        else:
            model_selection = jl.Parallel(n_jobs=-1, backend='threading') \
                (jl.delayed(tsa.ar_model.AutoReg(self.train_target.values,lag,self.trend).fit)() for lag in self.lags)
        
        # Pick best model based on the specified information criterion (AIC or BIC)
        ar_ics = [model.aic for model in model_selection] if self.ic=='aic' else [model.bic for model in model_selection]
        self.model = model_selection[ar_ics.index(min(ar_ics))]
        
    def forecast(self):
        # Make forecasts
        if self.exog:
            forecast = self.model.predict(start=self.train_target.shape[0], end=self.train_target.shape[0]+23, exog_oos=self.test_predictors)
        else:
            forecast = self.model.predict(start=self.train_target.shape[0], end=self.train_target.shape[0]+23)
        
        # Store forecasts in labelled dataframe
        forecast_df = pd.DataFrame(dict(Forecast=forecast), index=self.test_predictors.index)
        return(forecast_df)
    
############################################################################################################################################
# SARIMA/SARIMAX

class SARIMAX():
    """
    SARIMA/SARIMAX model implementation.
    
    Parameters
    ----------
        model_params : dict
            exog : bool
                Whether or not to include exogenous data (i.e. implement ARX (True) or AR (False))
            trend : str in ['n', 'c', 't', 'ct']
                Specifies ar_model.AutoReg() parameter for model trend, if any.
            order : tuple of length 3
                Specifies order of standard ARIMA component - (p,d,q).
            seasonal_order : tuple of length 4
                Specifies order of seasonal ARIMA component - (P,D,Q,S).
            method : str in ['newton', 'nm', 'bfgs', 'lbfgs', 'poewll', 'cg', 'ncg', 'basinhopping']
                Optimisation algorithm to use for estimating the model parameters.
            maxiter : int
                Maximum number of iterations of the given optimisation algorithm.
            disp : bool
                Whether to print progress logs under the optimisation algorithm.
        lag_params : dict
            bm_price_lags : list of int
                Lags of balancing market price data to use as predictors
            planned_lags : list of int
                Lags of forecast and wind data to use as predictors
    """
    def __init__(self, model_params, lag_params):
        self.exog = model_params['exog']
        self.trend = model_params['trend']
        self.order = model_params['order']
        self.seasonal_order = model_params['seasonal_order']
        self.method = model_params['method']
        self.maxiter = model_params['maxiter']
        self.disp = model_params['disp']
        
        self.bm_price_lags = lag_params['bm_price_lags']
        self.planned_lags = lag_params['planned_lags']
        
    def ingest_data(self, train_target, train_bm, train_planned, train_bid_curves):
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)

        # Split into training predictors and test predictors
        last_day_of_data = dt.datetime.combine(train_planned.index.date[-1], dt.datetime.min.time())
        test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq='H')
    
        # Initialise predictors dataframe
        predictors = pd.DataFrame(index=train_planned.index)
        
        # Build predictors from ImbalancePrice
        for lag in self.bm_price_lags:
            predictor_name = f"{train_bm.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_bm)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Build predictors from Wind and Demand
        for column in train_planned:
            for lag in self.planned_lags:
                predictor_name = f"{column}-{lag}"
                predictors.insert(predictors.shape[1], predictor_name, train_planned[column])
                predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Split predictors into training and test predictors and store data for training and forecasting
        train_predictors = predictors.drop(test_index)
        test_predictors = predictors.loc[test_index,:]
        
        # Remove rows in training set with NAs
        notna_train_predictors_loc = train_predictors.notna().all(axis=1)
        train_predictors = train_predictors.loc[notna_train_predictors_loc]
        train_target = train_target.loc[notna_train_predictors_loc]
        
        # Store data into object
        self.train_predictors = train_predictors
        self.test_predictors = test_predictors
        self.train_target = train_target
        
        # Initialise dataframe for variable importances
#         if not hasattr(self, "variable_importances"):
#             self.variable_importances = pd.DataFrame(index=train_predictors.columns)
        
    def train(self):
        # Fit AR/ARX models with different lag values
        if self.exog:
            self.model = tsa.statespace.sarimax.SARIMAX(endog=self.train_target,exog=self.train_predictors,
                                                        order=self.order,seasona_order=self.seasonal_order,
                                                        trend=self.trend).fit(method=self.method,maxiter=self.maxiter,disp=self.disp)
        else:
            self.model = tsa.statespace.sarimax.SARIMAX(endog=self.train_target,order=self.order,seasonal_order=self.seasonal_order,
                                                        trend=self.trend).fit(method=self.method,maxiter=self.maxiter,disp=self.disp)
        
    def forecast(self):
        # Make forecasts
        if self.exog:
            forecast = self.model.predict(start=self.train_predictors.shape[0], end=self.train_predictors.shape[0]+23, exog=self.test_predictors)
        else:
            forecast = self.model.predict(start=self.train_predictors.shape[0], end=self.train_predictors.shape[0]+23)
        forecast_df = pd.DataFrame(dict(Forecast=forecast), index=self.test_predictors.index)
        return(forecast_df)
    
############################################################################################################################################
# Artificial Neural Networks

def create_ffnn(num_of_nodes, input_cols, act_fn, n_layers=3, opt='adam', loss='mse'):
    """
    Create a Keras neural network model (composed of Dense feedforward layers).
    
    Parameters
    ----------
        num_of_nodes : int
            Number of neurons per layer.
        input_cols : int
            Number of features/predictors.
        act_fn : str in ['tanh', 'sigmoid', 'relu']
            Activation function.
        n_layers : int, default=3
            Number of hidden layers.
        opt : str, default='adam'
            Optimiser for training.
        loss : str, default='mse'
            Loss function for training.
    Returns
    -------
        model : tensorflow.python.keras.engine.sequential.Sequential
            Sequential model representing the neural network model.
    """
    # Initialise neural network model
    model = Sequential()
    initializer = initializers.he_uniform(1)
    
    # Add first hidden layer (with input layer specification)
    model.add(Dense(num_of_nodes, activation=act_fn, input_shape=(input_cols,), kernel_initializer=initializer))
    
    # Add remaining hidden layers
    for _ in range(n_layers-1):
        model.add(Dense(num_of_nodes, activation=act_fn, kernel_initializer=initializer))

    # Add output layer
    model.add(Dense(1, kernel_initializer=initializer))

    # Configure model optimizer and loss function
    model.compile(optimizer=opt, loss=loss)

    return(model)


def scale_predictors(predictors, activation, copy_df=True):
    """
    Rescale a dataframe of predictors according to the given activation function.
    
    Parameters
    ----------
        predictors : pandas.DataFrame
        activation : str in ['tanh', 'sigmoid', 'relu']
        
    Returns
    -------
        scaler : sklearn.preprocessing._data.MinMaxScaler
        scaled_predictors : pandas.DataFrame
    """
    activation_ranges = {
        'tanh': (-1,1),
        'sigmoid': (0,1),
        'relu': (0,5)
    }
    # Initialise scaler
    scaler = prep.MinMaxScaler(feature_range=activation_ranges[activation], copy=copy_df)
    
    # Fit scaler and scale the data
    scaled_predictors = pd.DataFrame(scaler.fit_transform(predictors), index=predictors.index, columns=predictors.columns)
    
    return(scaler, scaled_predictors)


class ffnn():
    """
    Feedforward neural network model implementation.
    
    Parameters
    ----------
        model_params : dict
            init_params : dict
                Arguments to pass into create_ffnn() function
            train_params : dict
                Extra arguments to pass into Sequential.fit() function
            other_params : dict
                Other extra arguments (number of epochs and bool to specify whether to use GPU)
        lag_params : dict
            price_lags : list of int
                Lags of day-ahead market price data to use as predictors
            bm_price_lags : list of int
                Lags of balancing market price data to use as predictors
            planned_lags : list of int
                Lags of forecast and wind data to use as predictors
    """
    def __init__(self, model_params, lag_params):
        self.init_params = model_params['init_params']
        self.train_params = model_params['train_params']
        self.other_params = model_params['other_params']
        
        # Set to GPU/CPU
        self.device = '/device:GPU:0' if self.other_params['GPU'] else '/CPU:0'
        
        if not hasattr(self, 'model'):
            self.model = create_ffnn(**self.init_params)
        
        self.price_lags = lag_params['price_lags']
        self.bm_price_lags = lag_params['bm_price_lags']
        self.planned_lags = lag_params['planned_lags']
        
    def ingest_data(self, train_target, train_bm, train_planned, train_bid_curves):
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)

        # Split into training predictors and test predictors
        last_day_of_data = dt.datetime.combine(train_planned.index.date[-1], dt.datetime.min.time())
        test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq='H')

        # Initialise predictors dataframe
        predictors = pd.DataFrame(index=train_planned.index)

        # Build predictors from EURPrices
        for lag in self.price_lags:
            predictor_name = f"{train_target.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_target)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Build predictors from ImbalancePrice
        for lag in self.bm_price_lags:
            predictor_name = f"{train_bm.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_bm)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)
            
        # Build predictors from Wind and Demand
        for column in train_planned:
            for lag in self.planned_lags:
                predictor_name = f"{column}-{lag}"
                predictors.insert(predictors.shape[1], predictor_name, train_planned[column])
                predictors[predictor_name] = predictors[predictor_name].shift(lag)
                
        # Split predictors into training and test predictors and store data for training and forecasting
        train_predictors = predictors.drop(test_index)
        test_predictors = predictors.loc[test_index,:]

        # Remove rows in training set with NAs
        notna_train_predictors_loc = train_predictors.notna().all(axis=1)
        train_predictors = train_predictors.loc[notna_train_predictors_loc]
        train_target = train_target.loc[notna_train_predictors_loc]

        # Scale predictors and target, and store in object
        self.train_predictors_scaler, self.scaled_train_predictors = scale_predictors(train_predictors, activation=self.init_params['act_fn'])
        self.train_target_scaler, self.scaled_train_target = scale_predictors(train_target, activation=self.init_params['act_fn'])
        self.scaled_test_predictors = self.train_predictors_scaler.transform(test_predictors)
        
        self.test_index = test_predictors.index
        
        # Initialise dataframe for variable importances
        if not hasattr(self, 'variable_importances'):
            self.variable_importances = pd.DataFrame(index=train_predictors.columns)

    def train(self):
        # Fit the forecasting model
        if hasattr(self, 'history'):
            self.history = self.model.fit(x=self.scaled_train_predictors.values, y=self.scaled_train_target.values, epochs=self.other_params['subseq_epochs'], batch_size=len(self.scaled_train_target), **self.train_params)
        else:
            self.history = self.model.fit(x=self.scaled_train_predictors.values, y=self.scaled_train_target.values, epochs=self.other_params['init_epochs'], batch_size=len(self.scaled_train_target), **self.train_params)
        
    def forecast(self):
        forecast = self.train_target_scaler.inverse_transform(self.model.predict(x=self.scaled_test_predictors))
        forecast_df = pd.DataFrame(forecast, index=self.test_index)
        return(forecast_df)
    
############################################################################################################################################
# Recurrent Neural Networks

def create_data_window(data, n_steps, step_size=24):
    """
    Given a (single column) predictor dataframe, returns a dataframe where each column is a lagged version
    of the original column. This is part of the pre-processing step for reformatting data as RNN input/s.
    
    Parameters
    ----------
        data : pandas.DataFrame
        n_steps : int
            Maximum number of steps to shift back. 
        step_size : int, default=24
    """
    new_data = data.copy()
    column = data.columns[0]

    # Add lagged values as new columns
    for step in range(n_steps):
        new_data.insert(new_data.shape[1], f"{column}-{step_size*(step+1)}h", new_data[[column]].shift(step_size * (step+1)))
    
    return(new_data)


def get_rnn_scaler(predictors, activation, copy_df=True):
    """
    Given a dataframe, return a scaler that can be used on other dataframes. This function is used
    for the predictors. The scaler is fit on (and used to transform) the train predictors and then
    used to transform the test predictors.
    
    Parameters
    ----------
        predictors : pandas.DataFrame
        activation : str in ['tanh', 'sigmoid', 'relu']
        copy_df : bool, default=True
            Set to False to perform inplace row normalization and avoid a
            copy (if the input is already a numpy array).
            
    Returns
    -------
        scaler : sklearn.preprocessing._data.MinMaxScaler
    """
    activation_ranges = {
        'tanh': (-1,1),
        'sigmoid': (0,1),
        'relu': (0,5)
    }
    real_predictors = predictors[:,0,:].copy()
    
    # Create and fit scaler
    scaler = prep.MinMaxScaler(feature_range=activation_ranges[activation], copy=copy_df)
    scaler.fit(real_predictors)
    
    return(scaler)


def transform_rnn_scale(predictors, scaler):
    """
    Function to apply the scaler to the array of predictors. This function is needed instead
    of the usual scaler.transform() method because predictors is a 3D numpy array instead
    of the 2D dataframe/array expected by the function.
    
    Parameters
    ----------
        predictors : numpy.array
        scaler : sklearn.preprocessing._data.MinMaxScaler
        
    Returns
    -------
        scaled_predictors : numpy.array
    """
    # Transform predictors
    scaled_predictors = predictors.copy()
    
    for time_step in range(scaled_predictors.shape[1]):
        scaled_predictors[:,time_step,:] = scaler.transform(scaled_predictors[:,time_step,:])
        
    return(scaled_predictors)


def create_rnn(num_of_blocks, n_timesteps, n_features, act_fn, n_layers=3, opt='adam', loss='mse'):
    """
    Create a Keras neural network model (composed of LSTM layers).
    
    Parameters
    ----------
        num_of_blocks : int
            Number of neurons per layer.
        input_cols : int
            Number of features/predictors (excluding lagged predictors).
        act_fn : str in ['tanh', 'sigmoid', 'relu']
            Activation function.
        n_layers : int, default=3
            Number of hidden layers.
        opt : str, default='adam'
            Optimiser for training.
        loss : str, default='mse'
            Loss function for training.
    Returns
    -------
        model : tensorflow.python.keras.engine.sequential.Sequential
            Sequential model representing the neural network model.
    """
    model = Sequential()

    # Add first hidden layer
    if n_layers == 1:
        model.add(LSTM(num_of_blocks, activation=act_fn, input_shape=(n_timesteps, n_features)))
    else:
        model.add(LSTM(num_of_blocks, return_sequences=True, activation=act_fn, input_shape=(n_timesteps, n_features)))
    
    # Add remaining hidden layers
    for n in range(n_layers-1):
        if n == n_layers-2:
            model.add(LSTM(num_of_blocks, activation=act_fn))
        else:
            model.add(LSTM(num_of_blocks, return_sequences=True, activation=act_fn))

    # Add output layer
    model.add(Dense(1))#, kernel_initializer=initializer))

    # Configure model optimizer and loss function
    model.compile(optimizer=opt, loss=loss)

    return(model)


class rnn():
    """
    Recurrent neural network (LSTM) model implementation.
    
    Parameters
    ----------
        model_params : dict
            init_params : dict
                Arguments to pass into create_rnn() function
            train_params : dict
                Extra arguments to pass into Sequential.fit() function
            other_params : dict
                Other extra arguments (number of epochs and bool to specify whether to use GPU)
    """
    def __init__(self, model_params, lag_params):
        self.init_params = model_params['init_params']
        self.train_params = model_params['train_params']
        self.other_params = model_params['other_params']
        
        # Set to GPU/CPU
        self.device = '/device:GPU:0' if self.other_params['GPU'] else '/CPU:0'
        
        if not hasattr(self, 'model'):
            self.model = create_rnn(**self.init_params)
        
    def ingest_data(self, train_target, train_bm, train_planned, train_bid_curves):
        # Get the the latest start date of the variables
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)
        target_end_date = train_target.index[-1]

        # Split up planned data into its component columns
        train_wind = train_planned.loc[:,'Wind'].to_frame()
        train_demand = train_planned.loc[:,'Demand'].to_frame()
        
        # Add extra rows corresponding to forecast dates
        empty_frame = pd.DataFrame(index=train_planned.index[-24:])
        train_target = train_target.append(empty_frame)
        train_bm = train_bm.append(empty_frame)
        
        # Reformat data for input into RNNs
        n_timesteps = self.init_params['n_timesteps']
        rnn_train_prices = create_data_window(train_target, n_timesteps)

        rnn_train_target = rnn_train_prices[[rnn_train_prices.columns[0]]]
        rnn_train_prices.drop(rnn_train_prices.columns[0], axis=1, inplace=True)

        rnn_train_bm = create_data_window(train_bm, n_timesteps)
        rnn_train_bm.drop(rnn_train_bm.columns[0], axis=1, inplace=True)

        rnn_train_wind = create_data_window(train_wind, n_timesteps)
        rnn_train_wind.drop(rnn_train_wind.columns[-1], axis=1, inplace=True)

        rnn_train_demand = create_data_window(train_demand, n_timesteps)
        rnn_train_demand.drop(rnn_train_demand.columns[-1], axis=1, inplace=True)
        
        # Remove all rows with NAs that arose due to the reformatting above
        rnn_train_prices = rnn_train_prices.loc[rnn_train_prices.notna().all(axis=1)]
        rnn_train_target = rnn_train_target.loc[rnn_train_target.notna().all(axis=1)]
        rnn_train_bm = rnn_train_bm.loc[rnn_train_bm.notna().all(axis=1)]
        rnn_train_wind = rnn_train_wind.loc[rnn_train_wind.notna().all(axis=1)]
        rnn_train_demand = rnn_train_demand.loc[rnn_train_demand.notna().all(axis=1)]
        
        # Match data start dates (again)
        rnn_train_target = rnn_train_target.loc[rnn_train_target.index >= rnn_train_prices.index[0]]
        rnn_train_bm = rnn_train_bm.loc[rnn_train_bm.index >= rnn_train_prices.index[0]]
        rnn_train_wind = rnn_train_wind.loc[rnn_train_wind.index >= rnn_train_prices.index[0]]
        rnn_train_demand = rnn_train_demand.loc[rnn_train_demand.index >= rnn_train_prices.index[0]]
        
        # Split into training and test set
        self.test_index = rnn_train_prices.index[-24:]

        # Test data
        rnn_test_prices = rnn_train_prices.loc[rnn_train_prices.index >= self.test_index[0]]
        rnn_test_bm = rnn_train_bm.loc[rnn_train_bm.index >= self.test_index[0]]
        rnn_test_wind = rnn_train_wind.loc[rnn_train_wind.index >= self.test_index[0]]
        rnn_test_demand = rnn_train_demand.loc[rnn_train_demand.index >= self.test_index[0]]

        # Train data
        rnn_train_prices = rnn_train_prices.loc[rnn_train_prices.index < self.test_index[0]]
        rnn_train_bm = rnn_train_bm.loc[rnn_train_bm.index < self.test_index[0]]
        rnn_train_wind = rnn_train_wind.loc[rnn_train_wind.index < self.test_index[0]]
        rnn_train_demand = rnn_train_demand.loc[rnn_train_demand.index < self.test_index[0]]
        rnn_train_target = rnn_train_target.loc[rnn_train_target.index < self.test_index[0]]
        
        # Combine train and test features into one tensor each
        rnn_test_predictors = np.hstack((rnn_test_prices, rnn_test_bm, rnn_test_wind, rnn_test_demand)).reshape(rnn_test_prices.shape[0], 4, n_timesteps).transpose(0,2,1)
        rnn_train_predictors = np.hstack((rnn_train_prices, rnn_train_bm, rnn_train_wind, rnn_train_demand)).reshape(rnn_train_prices.shape[0], 4, n_timesteps).transpose(0,2,1)

        # Scale predictors and target, and store in object
        self.rnn_train_predictors_scaler = get_rnn_scaler(rnn_train_predictors, activation=self.init_params["act_fn"])
        self.rnn_scaled_train_predictors = transform_rnn_scale(rnn_train_predictors, self.rnn_train_predictors_scaler)
        self.rnn_scaled_test_predictors = transform_rnn_scale(rnn_test_predictors, self.rnn_train_predictors_scaler)
        self.rnn_train_target_scaler, self.rnn_scaled_train_target = scale_predictors(rnn_train_target, activation=self.init_params['act_fn'])


    def train(self):
        # Fit the forecasting model
        if hasattr(self, 'history'):
            with tf.device(self.device):
                self.history = self.model.fit(x=self.rnn_scaled_train_predictors, y=self.rnn_scaled_train_target.values, epochs=self.other_params['subseq_epochs'], **self.train_params)
        else:
            with tf.device(self.device):
                self.history = self.model.fit(x=self.rnn_scaled_train_predictors, y=self.rnn_scaled_train_target.values, epochs=self.other_params['init_epochs'], **self.train_params)

    def forecast(self):
        forecast = self.rnn_train_target_scaler.inverse_transform(self.model.predict(x=self.rnn_scaled_test_predictors))
        forecast_df = pd.DataFrame(forecast, index=self.test_index)
        return(forecast_df)
    
############################################################################################################################################
# X-Model
"""
"""

def get_price_curves(dataframe, index, return_vals="both", aggregated=False, plot=False, xlim=None, ylim=None, legend=True, figsize=mpl.rcParams["figure.figsize"]):
    # Extract bid volumes and corresponding bid prices for that delivery hour
    pv = list(map(float,dataframe.loc[:,"PurchaseVolume"][index].split(",")))
    sv = list(map(float,dataframe.loc[:,"SellVolume"][index].split(",")))
    pp = list(map(float, dataframe.loc[:,"PurchasePrice"][index].split(",")))
    sp = list(map(float,dataframe.loc[:,"SellPrice"][index].split(",")))
    
    # Combine bid data and supply data into their own dataframes, and add a column of aggregated volume to each df.
    if return_vals == "purchase":
        purchase = pd.DataFrame(dict(PurchaseVolume=pv, PurchasePrice=pp)).sort_values(by="PurchasePrice", ascending=False)
        if aggregated:
            purchase["AggregatedPurchaseVolume"] = purchase[["PurchaseVolume"]].cumsum()
        
    elif return_vals == "sell":
        sell = pd.DataFrame(dict(SellVolume=sv, SellPrice=sp)).sort_values(by="SellPrice")
        if aggregated:
            sell["AggregatedSellVolume"] = sell[["SellVolume"]].cumsum()
        
    elif return_vals == "both":
        purchase = pd.DataFrame(dict(PurchaseVolume=pv, PurchasePrice=pp)).sort_values(by="PurchasePrice", ascending=False)
        sell = pd.DataFrame(dict(SellVolume=sv, SellPrice=sp)).sort_values(by="SellPrice")
        
        if aggregated:
            purchase["AggregatedPurchaseVolume"] = purchase[["PurchaseVolume"]].cumsum()
            sell["AggregatedSellVolume"] = sell[["SellVolume"]].cumsum()
    
    # For illustration
    if plot and aggregated:
        xlim = (purchase["AggregatedPurchaseVolume"].min(),
                purchase["AggregatedPurchaseVolume"].max()) if xlim==None else xlim
        ylim = (-100, 300) if ylim==None else ylim
        ax = purchase.drop(["PurchaseVolume"], axis=1).plot(x="AggregatedPurchaseVolume", legend=legend, figsize=figsize)
        sell.drop(["SellVolume"], axis=1).plot(x="AggregatedSellVolume", ax=ax, legend=legend)
        ax.set_xlabel("Volume")
        ax.set_ylabel("Price")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    if plot and not aggregated:
        print("Warning: cannot plot if aggregated=False")
           
    # Return the purchase/sell curve data
    if return_vals == "purchase":
        return(purchase) 
    elif return_vals == "sell":
        return(sell)
    elif return_vals == "both":
        return(purchase, sell)


def get_mean_volumes(dataframe):
    n = dataframe.shape[0]
    
    # Create dataframes for combining hourly purchase and sell bids
    combined_purchase = pd.DataFrame(columns=["PurchaseVolume", "PurchasePrice"])
    combined_sell = pd.DataFrame(columns=["SellVolume", "SellPrice"])
    
    # Combine the bids
    for i in range(n):
        hourly_purchase, hourly_sell = get_price_curves(dataframe, i)
        combined_purchase = combined_purchase.append(hourly_purchase, ignore_index=True)
        combined_sell = combined_sell.append(hourly_sell, ignore_index=True)
    
    # Calculate the mean volumes across each distinct price. Note: we divide by n rather than using .mean() function
    # since there are many times when a volume for a given price is 0. The .mean() function does not take this into account,
    # and we want to include the 0 values in the calculation of the mean.
    mean_purchase = combined_purchase.groupby(["PurchasePrice"]).sum()/n
    mean_sell = combined_sell.groupby(["SellPrice"]).sum()/n
    
    return(mean_purchase, mean_sell)


def to_price_curve(bid_dataframe, curve_type):
    # Sort dataframe by prices (increasing order for supply curve, decreasing order for demand curve)
    if curve_type == "sell":
        sorted_bid_dataframe = bid_dataframe.sort_values(by=["SellPrice"], ascending=True)
    elif curve_type == "purchase":
        sorted_bid_dataframe = bid_dataframe.sort_values(by=["PurchasePrice"], ascending=False)

    # Calculate cumulative volumes
    column_name = "PurchaseVolume" if curve_type == "purchase" else "SellVolume"
    agg_column_name = "AggregatedPurchaseVolume" if curve_type == "purchase" else "AggregatedSellVolume"
    
    sorted_bid_dataframe[agg_column_name] = sorted_bid_dataframe[column_name].cumsum()
    
    return(sorted_bid_dataframe)


""" Each purchase (demand) curve price class is determined by its lower bound, and
    each sell (supply) curve price class is determined by its upper bound.
    
    The class_length parameter can be chosen arbitrarily or with some justification. Our choice of 31000 is to so
    that we have a similar number of classes as in the paper (which had 16 purchase/sell classes).
"""
def create_price_classes(mean_purchase_curve, mean_sell_curve, class_length=35000, decimals=2):
    # Initialise and prepare price class dataframes.
    purchase_df = mean_purchase_curve.drop("PurchaseVolume", axis=1)
    sell_df = mean_sell_curve.drop("SellVolume", axis=1)

    purchase_classes = pd.DataFrame(columns=["Volume", "Price"])
    sell_classes = pd.DataFrame(columns=["Volume","Price"])
    
    purchase_classes = purchase_classes.append(dict(Volume=purchase_df.iloc[0,0], Price=purchase_df.index[0]), ignore_index=True)
    sell_classes = sell_classes.append(dict(Volume=sell_df.iloc[0,0], Price=sell_df.index[0]), ignore_index=True)
    
    
    # Populate demand price class dataframe
    volume_class_increment = 1
    for i in range(1, purchase_df.shape[0]):
        new_volume_class_bound = volume_class_increment * class_length
        
        if new_volume_class_bound == purchase_df.iloc[i, 0]:
            purchase_classes = purchase_classes.append(dict(Volume=new_volume_class_bound, Price=purchase_df.index[i]))
        
        elif new_volume_class_bound < purchase_df.iloc[i, 0]:
            previous_price = purchase_df.index[i-1]
            current_price = purchase_df.index[i]
            previous_volume = purchase_df.iloc[i-1, 0]
            current_volume = purchase_df.iloc[i, 0]
            
            # Linear interpolation between current price and previous price
            price_diff = current_price - previous_price
            volume_diff = current_volume - previous_volume
            new_price_class_bound = previous_price + (price_diff/volume_diff)*(new_volume_class_bound-previous_volume)
            purchase_classes = purchase_classes.append(dict(Volume=new_volume_class_bound, Price=round(new_price_class_bound, decimals)), ignore_index=True)
        else:
            continue
            
        volume_class_increment += 1
    
    
    # Populate supply price class dataframe
    volume_class_increment = 1
    for i in range(1, sell_df.shape[0]):
        new_volume_class_bound = volume_class_increment * class_length
        
        if new_volume_class_bound == sell_df.iloc[i, 0]:
            sell_classes = sell_classes.append(dict(Volume=new_volume_class_bound, Price=sell_df.index[i]), ignore_index=True)
        
        elif new_volume_class_bound < sell_df.iloc[i, 0]:
            previous_price = sell_df.index[i-1]
            current_price = sell_df.index[i]
            previous_volume = sell_df.iloc[i-1, 0]
            current_volume = sell_df.iloc[i, 0]
            
            # Linear interpolation between current price and previous price
            price_diff = current_price - previous_price
            volume_diff = current_volume - previous_volume
            new_price_class_bound = previous_price + (price_diff/volume_diff)*(new_volume_class_bound-previous_volume)
            sell_classes = sell_classes.append(dict(Volume=new_volume_class_bound, Price=round(new_price_class_bound, decimals)), ignore_index=True)
        else:
            continue
            
        volume_class_increment += 1
    
    purchase_classes = purchase_classes.append(dict(Volume=purchase_df.iloc[purchase_df.shape[0]-1,0], Price=round(purchase_df.index[-1], decimals)), ignore_index=True)
    sell_classes = sell_classes.append(dict(Volume=sell_df.iloc[sell_df.shape[0]-1,0], Price=round(sell_df.index[-1], decimals)), ignore_index=True)
    
    return(purchase_classes, sell_classes)


def reduce_bid_ask_data(bid_ask_dataframe, purchase_classes, sell_classes):
    # Initialise and prepare dataframes
    purchase_columns = list(purchase_classes["Price"])
    sell_columns = list(sell_classes["Price"])
    
    purchase_reduced_df = pd.DataFrame(columns=purchase_columns, index=bid_ask_dataframe.index).fillna(0)
    sell_reduced_df = pd.DataFrame(columns=sell_columns, index=bid_ask_dataframe.index).fillna(0)
    
    # Populate price classes with bid volumes
    for time_step in range(bid_ask_dataframe.shape[0]):
        index = bid_ask_dataframe.index[time_step]
        purchase_df, sell_df = get_price_curves(bid_ask_dataframe, time_step)
        purchase_df = purchase_df.reset_index(drop=True)
        
        purchase_classes_index = sell_classes_index = 0
        
        current_purchase_class = purchase_classes.loc[purchase_classes_index, "Price"]
        for i in range(purchase_df.shape[0]):
            current_purchase_price = purchase_df.loc[i, "PurchasePrice"]
            
            while(current_purchase_class > current_purchase_price):
                purchase_classes_index += 1
                current_purchase_class = purchase_classes.loc[purchase_classes_index, "Price"]
                
            purchase_reduced_df.iloc[time_step, purchase_classes_index] += purchase_df.loc[i, "PurchaseVolume"]
            
            
        current_sell_class = sell_classes.loc[sell_classes_index, "Price"]
        for j in range(sell_df.shape[0]):
            current_sell_price = sell_df.loc[j, "SellPrice"]
            
            while(current_sell_class < current_sell_price):
                sell_classes_index += 1
                current_sell_class = sell_classes.loc[sell_classes_index, "Price"]
            
            sell_reduced_df.iloc[time_step, sell_classes_index] += sell_df.loc[j, "SellVolume"]
            
    return(purchase_reduced_df, sell_reduced_df)


# Function to find the specific hour (TimeStepID) that is either duplicated (for 25-hour days) or is missing (for 23-hour days).
# The usual DST is for hour 3, i.e. either two rows for hour 3, or no hour 3 at all, but we make the function a bit more flexible.
def find_dst_index(time_step_id_dataframe, number_of_hours):
    if number_of_hours == 23:
        for i, time_step_id in enumerate(time_step_id_dataframe):
            if i < time_step_id:
                return(i)
            elif i == number_of_hours:
                return(23)
                
    elif number_of_hours == 25:
        for j, time_step_id in enumerate(time_step_id_dataframe):
            if j > time_step_id:
                return(time_step_id)

# Function to ensure that bid data consists of 24 hours per day. For 23-hour days, the extra day is added by interpolating
# the bid data between the previous and next hour. For 25-hour days, we replace the duplicate hours by getting the average.
def dst_adjustment(reduced_bid_df):
    reduced_bid_df = reduced_bid_df.copy()
    
    # Fetch the list of DST days - If we did our pre-processing correctly in previous sections,
    # this dataframe should only have values of 23 and 25.
    df_count = reduced_bid_df.groupby([reduced_bid_df.index.date]).count()
    dst_dates = df_count.loc[(df_count != 24).any(axis=1)][-500]
    
    if dst_dates.shape[0] == 0:
        return(reduced_bid_df)
    
    for i in range(dst_dates.shape[0]):
        dst_date = dst_dates.index[i]
        number_of_hours = dst_dates[i]
        
        # Get the (reduced) bid data for the specific dst_date
        reduced_bid_dst_data = reduced_bid_df.loc[reduced_bid_df.index.date == dst_date,:]

        # Find the specific hour that's either duplicated (for 25-hour days) or missing (for 23-hour days).
        dst_index = find_dst_index(reduced_bid_dst_data.index.hour, number_of_hours)

        # If 23-hour day, get the average of the bid data for the adjacent hours.
        if number_of_hours == 23:
            # Fetch adjacent bids, e.g. if missing bid is for 3rd hour, then we fetch bids for 2nd and 4th hour.
            previous_bid = reduced_bid_dst_data.loc[reduced_bid_dst_data.index.hour == dst_index-1]
            next_bid = reduced_bid_dst_data.loc[reduced_bid_dst_data.index.hour == dst_index+1]
            adjacent_bids = previous_bid.append(next_bid)
            
            # Calculate the average of the two hours of bid data
            average_values = adjacent_bids.mean(axis=0).rename(reduced_bid_df.shape[0]-1).to_frame().T
            average_values.index = pd.DatetimeIndex([dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            
            reduced_bid_df = reduced_bid_df.append(average_values)
        
        # If 25-hour day, replace the two duplicate hours with their average
        elif number_of_hours == 25:
            # Fetch duplicate bids
            duplicate_bids = reduced_bid_dst_data.loc[reduced_bid_dst_data.index.hour == dst_index]
            
            # Calculate the average of the two hours of bid data
            average_values = duplicate_bids.mean(axis=0).rename(reduced_bid_df.shape[0]-1).to_frame().T
            average_values.index = pd.DatetimeIndex([dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])

            # Delete the two rows of duplicate hours
            reduced_bid_df.drop(duplicate_bids.index, inplace=True)
        
            # Insert this new bid into the original dataframe containing reduced bid data.
            reduced_bid_df = reduced_bid_df.append(average_values)
            

    # Sort the dataframe so that the bid data is properly arranged by DeliveryPeriod, and rename index
    reduced_bid_df.sort_index(axis=0, inplace=True)
    reduced_bid_df.index.name = "DeliveryPeriod"
        
    return(reduced_bid_df)


# Function to find the specific hour (TimeStepID) that is either duplicated (for 25-hour days) or is missing (for 23-hour days).
# The usual DST is for hour 3, i.e. either two rows for hour 3, or no hour 3 at all, but we make the function a bit more flexible.
def find_dst_index(time_step_id_dataframe, number_of_hours):
    if number_of_hours == 23:
        for i, time_step_id in enumerate(time_step_id_dataframe):
            if i < time_step_id:
                return(i)
            elif i == number_of_hours:
                return(23)
                
    elif number_of_hours == 25:
        for j, time_step_id in enumerate(time_step_id_dataframe):
            if j > time_step_id:
                return(time_step_id)

# Function to ensure that bid data consists of 24 hours per day. For 23-hour days, the extra day is added by interpolating
# the bid data between the previous and next hour. For 25-hour days, we replace the duplicate hours by getting the average.
def dst_adjustment(reduced_bid_df):
    reduced_bid_df = reduced_bid_df.copy()
    
    # Fetch the list of DST days - If we did our pre-processing correctly in previous sections,
    # this dataframe should only have values of 23 and 25.
    df_count = reduced_bid_df.groupby([reduced_bid_df.index.date]).count()
    dst_dates = df_count.loc[(df_count != 24).any(axis=1)][-500]
    
    if dst_dates.shape[0] == 0:
        return(reduced_bid_df)
    
    for i in range(dst_dates.shape[0]):
        dst_date = dst_dates.index[i]
        number_of_hours = dst_dates[i]
        
        # Get the (reduced) bid data for the specific dst_date
        reduced_bid_dst_data = reduced_bid_df.loc[reduced_bid_df.index.date == dst_date,:]

        # Find the specific hour that's either duplicated (for 25-hour days) or missing (for 23-hour days).
        dst_index = find_dst_index(reduced_bid_dst_data.index.hour, number_of_hours)

        # If 23-hour day, get the average of the bid data for the adjacent hours.
        if number_of_hours == 23:
            # Fetch adjacent bids, e.g. if missing bid is for 3rd hour, then we fetch bids for 2nd and 4th hour.
            previous_bid = reduced_bid_dst_data.loc[reduced_bid_dst_data.index.hour == dst_index-1]
            next_bid = reduced_bid_dst_data.loc[reduced_bid_dst_data.index.hour == dst_index+1]
            adjacent_bids = previous_bid.append(next_bid)
            
            # Calculate the average of the two hours of bid data
            average_values = adjacent_bids.mean(axis=0).rename(reduced_bid_df.shape[0]-1).to_frame().T
            average_values.index = pd.DatetimeIndex([dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            
            reduced_bid_df = reduced_bid_df.append(average_values)
        
        # If 25-hour day, replace the two duplicate hours with their average
        elif number_of_hours == 25:
            # Fetch duplicate bids
            duplicate_bids = reduced_bid_dst_data.loc[reduced_bid_dst_data.index.hour == dst_index]
            
            # Calculate the average of the two hours of bid data
            average_values = duplicate_bids.mean(axis=0).rename(reduced_bid_df.shape[0]-1).to_frame().T
            average_values.index = pd.DatetimeIndex([dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])

            # Delete the two rows of duplicate hours
            reduced_bid_df.drop(duplicate_bids.index, inplace=True)
        
            # Insert this new bid into the original dataframe containing reduced bid data.
            reduced_bid_df = reduced_bid_df.append(average_values)
            

    # Sort the dataframe so that the bid data is properly arranged by DeliveryPeriod, and rename index
    reduced_bid_df.sort_index(axis=0, inplace=True)
    reduced_bid_df.index.name = "DeliveryPeriod"
        
    return(reduced_bid_df)

""" This is for the issue of the bid curve data missing entire day of data for some specific days.
"""
def clean_reduced_bid_data(dataframe):
    df = dataframe.copy()
    
    # Insert missing dates (as NaNs) into the dataframe
    start_date = df.index[0]
    end_date = df.index[-1]
    df = df.reindex(pd.date_range(start_date, end_date, freq='H'))
    
    # Calculate overall mean for each given hour of each given day (Monday to Sunday)
    df["DayOfWeek"] = df.index.dayofweek
    means = df.groupby(["DayOfWeek", df.index.hour]).mean()
    
    # Fetch list of missing dates
    missing_dates = df.loc[df.isna().any(axis=1)].index

    # Replace missing values with appropriate mean value
    for date in missing_dates:
        day_of_week = df["DayOfWeek"].loc[date]
        hour = date.hour
        mean = means.loc[day_of_week, hour]
        df.loc[date] = mean
    
    return(df.drop("DayOfWeek", axis=1))


# Function to determine the number of lagged versions of a given predictor to set as dependencies of the target variable
def get_lags(lag_structure, target_class, target_hour, predictor_class, predictor_hour):
    same_class = target_class == predictor_class
    same_hour = target_hour == predictor_hour
    
    if same_class and same_hour:
        return(lag_structure["same_hc"])
    elif same_class and not same_hour:
        return(lag_structure["same_c"])
    elif not same_class and same_hour:
        return(lag_structure["same_h"])
    else:
        return(lag_structure["other"])
    

def build_XM_predictors(reduced_purchase, reduced_sell, exog_classes, planned_classes, lag_structure, verbose=False):
    purchase = reduced_purchase.copy()
    sell = reduced_sell.copy()
    
    # Rename purchase and sell price classes for distinguishing between the two
    purchase.columns = pd.Index([str(column)+"P" for column in reduced_purchase.columns])
    sell.columns = pd.Index([str(column)+"S" for column in reduced_sell.columns])

    # Combine purchase/sell classes into one
    price_classes = purchase.join(sell, how="outer")
    
    if verbose: print("Building _variables_df dataframes")
    build_start = time.time()
    
    train_variables_df = pd.DataFrame(index=pd.unique(price_classes.index.date))
    forecast_variables_df = pd.DataFrame(index=[price_classes.index.date[-1] + dt.timedelta(days=1)])
    
    days_of_the_week = {"Monday":0, "Tuesday":1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5}
    
    # Add day of the week indicators
    for key, value in days_of_the_week.items():
        train_dotw = list(map(lambda val: int(val <= value), pd.DatetimeIndex(train_variables_df.index).dayofweek.tolist()))
        forecast_dotw = list(map(lambda val: int(val <= value), pd.DatetimeIndex(forecast_variables_df.index).dayofweek.tolist()))
        
        train_variables_df.insert(value, key, train_dotw)
        forecast_variables_df.insert(value, key, forecast_dotw)
    
    for hour in range(24):
        if verbose: print("Building hour", hour)
        
        # Add price classes as predictors
        for price_class in price_classes.columns:
            max_lags = list(range(1,max([len(lags) for lags in lag_structure.values()]) + 1))
            for lag in max_lags:
                variable = price_classes.loc[price_classes.index.hour==hour, price_class]
                variable_name = f"{price_class}_H{hour}-{lag}"
                
                train_variable = variable.shift(lag)
                train_variable.index = train_variable.index.date
                train_variables_df.insert(train_variables_df.shape[1], variable_name, train_variable)
                
                forecast_variable = variable[-lag]
                forecast_variables_df.insert(forecast_variables_df.shape[1], variable_name, forecast_variable)
        
        # Add exogenous classes (prices and BM prices) as predictors
        for exog_class in exog_classes.columns:
            max_lags = lag_structure["same_h"]
            for lag in max_lags:
                variable = exog_classes.loc[exog_classes.index.hour==hour, exog_class]
                variable_name = f"{exog_class}_H{hour}-{lag}"
                
                train_variable = variable.shift(lag)
                train_variable.index = train_variable.index.date
                train_variables_df.insert(train_variables_df.shape[1], variable_name, train_variable)
                
                forecast_variable = variable[-lag]
                forecast_variables_df.insert(forecast_variables_df.shape[1], variable_name, forecast_variable)

        
        # Add planed classes (wind and demand forecast) as predictors
        for planned_class in planned_classes.columns:
            max_lags = list(map(lambda lag: lag-1, lag_structure["same_h"]))
            for lag in max_lags:
                variable = planned_classes.loc[planned_classes.index.hour==hour, planned_class]
                variable_name = f"{planned_class}_H{hour}-{lag}"
                
                train_variable = variable.shift(lag)
                train_variable.index = train_variable.index.date
                train_variables_df.insert(train_variables_df.shape[1], variable_name, train_variable)
                
                forecast_variable = variable[-lag-1]
                forecast_variables_df.insert(forecast_variables_df.shape[1], variable_name, forecast_variable)
                
    build_end = time.time()
    if verbose: print(f"_variables_df build execution time: {build_end-build_start} seconds.")
        
    price_classes.index.name = "DeliveryPeriod"
    train_variables_df.index.name = "DeliveryPeriod"
    forecast_variables_df.index.name = "DeliveryPeriod"
        
    return(price_classes, train_variables_df)
    return(price_classes, train_variables_df, forecast_variables_df)


def calculate_intersection(px1, py1, px2, py2, sx1, sy1, sx2, sy2):
    # Calculate slopes
    sell_m = (sy2 - sy1) / (sx2 - sx1)
    purchase_m = (py2 - py1) / (px2 - px1)
    
    # Calculate intersection points
    x = (sy1 - py1 + purchase_m*px1 - sell_m*sx1) / (purchase_m-sell_m)
    y = purchase_m*(x-px1) + py1
    return(x,y)


def find_bid_intersection(purchase, sell, verbose=False):
    purchase_index = 1; sell_index = 1
    
    for purchase_index in range(1, purchase.shape[0]):
        for sell_index in range(1, sell.shape[0]):
        # Relabel points
            pp1 = purchase.index[purchase_index];   pv1 = purchase.values[purchase_index]
            pp2 = purchase.index[purchase_index-1]; pv2 = purchase.values[purchase_index-1]

            sp1 = sell.index[sell_index];   sv1 = sell.values[sell_index]
            sp2 = sell.index[sell_index-1]; sv2 = sell.values[sell_index-1]

            # Check if one of the price points in the supply curve is between the two other points on the bid curve, or vice versa.
            # If so, then the intersection could potentially occur between these points.
            if (sp1 <= pp1 and sp1 >= pp2) or (sp2 <= pp1 and sp2 >= pp2) or (pp1 <= sp1 and pp1 >= sp2) or (pp2 <= sp1 and pp2 >= sp2):
                # Calculate intersection of lines formed by the pairs of points.
                # purchase: (pp1,pv1) & (pp2,pv2);     sell: (sp1,sp2) & (sp2,sv2)
                price, volume = calculate_intersection(pp1, pv1, pp2, pv2, sp1, sv1, sp2, sv2)

                # Determine if this intersection is between all the points. If so, then this intersection is the price forecast.
                if (price<pp1 and price>pp2 and price<sp1 and price>sp2) and (volume>pv1 and volume<pv2 and volume<sv1 and volume>sv2):
                    return(price, volume)


def forecast_prices_from_classes(forecast, verbose=False):
    purchase_classes = forecast.filter(regex="P")
    purchase_classes.columns = list(map(lambda string: float(string[:-1]), purchase_classes.columns))
    purchase_classes.sort_index(axis=1, ascending=False)

    sell_classes = forecast.filter(regex="S")
    sell_classes.columns = list(map(lambda string: float(string[:-1]), sell_classes.columns))
    sell_classes.sort_index(axis=1, ascending=True)
    
    purchase_classes_agg = purchase_classes.cumsum(axis=1)
    sell_classes_agg = sell_classes.cumsum(axis=1)

    forecast_df = pd.DataFrame(index=purchase_classes_agg.index, columns=["EURPrices"])
    
    for datetime in forecast_df.index:
        if verbose: print(datetime)
        hourly_purchase = purchase_classes_agg.loc[datetime].sort_index(ascending=True)
        hourly_sell = sell_classes_agg.loc[datetime]
        price, _ = find_bid_intersection(hourly_purchase, hourly_sell, verbose)
        forecast_df.loc[datetime] = price
    
    return(forecast_df)


def estimate_bid_probabilities(dataframe, price_curve="both"):
    n = dataframe.shape[0]
    
    # Create dataframes for combining hourly purchase and sell bids
    combined_purchase = pd.DataFrame(columns=["PurchasePrice"])
    combined_sell = pd.DataFrame(columns=["SellPrice"])
    
    # Combine the bids
    for i in range(n):
        hourly_purchase, hourly_sell = get_price_curves(dataframe, i)
        hourly_purchase = pd.DataFrame(pd.unique(hourly_purchase["PurchasePrice"]), columns=["PurchasePrice"]).reset_index()
        hourly_sell = pd.DataFrame(pd.unique(hourly_sell["SellPrice"]), columns=["SellPrice"]).reset_index()
        
        combined_purchase = combined_purchase.append(hourly_purchase, ignore_index=True)
        combined_sell = combined_sell.append(hourly_sell, ignore_index=True)

    # Calculate the mean volumes across each distinct price. Note: we divide by n rather than using .mean() function
    # since there are many times when a volume for a given price is 0. The .mean() function does not take this into account,
    # and we want to include the 0 values in the calculation of the mean.
    mean_purchase = combined_purchase.groupby(["PurchasePrice"]).count()/n
    mean_sell = combined_sell.groupby(["SellPrice"]).count()/n
    
    return(mean_purchase, mean_sell)


def round_bid_volume_prices(mean_volumes):
    mv = mean_volumes.copy()
    mv.index = [round(price, 1) for price in mv.index]
    mv.index.name = mean_volumes.index.name
    mv = mv.groupby(mv.index.name).sum()
    mv = mv.reindex(index=pd.Index([round(num,1) for num in np.arange(-500.,3000.1,0.1)]), fill_value=0, copy=False)

    return(mv)


def round_bid_volume_probabilities(mean_volumes):
    mv = mean_volumes.copy()
    mv.index = [round(price, 1) for price in mv.index]
    mv.index.name = mean_volumes.index.name
    mv = mv.groupby(mv.index).mean()
    mv = mv.reindex(index=pd.Index([round(num,1) for num in np.arange(-500.,3000.1,0.1)]), fill_value=0, copy=False)

    return(mv)


def reconstruct_bids(class_forecasts, mean_purchase_volumes, mean_sell_volumes, mean_purchase_probs, mean_sell_probs, threshold):
    # Convert probabilities to 1 if greater than a certain threshold, and 0 if less than
    purchase_binom = mean_purchase_probs.apply(lambda p: [int(val>threshold) for val in p])
    sell_binom = mean_sell_probs.apply(lambda p: [int(val>threshold) for val in p])
    
    # Keep only the rows for which the prices were reduced to 1, and drop all that were reduced to 0
    purchase_binom = purchase_binom.loc[purchase_binom["index"] == 1]
    sell_binom = sell_binom.loc[sell_binom["index"] == 1]
    
    purchase_active_bids = mean_purchase_volumes.copy().loc[purchase_binom.index]
    sell_active_bids = mean_sell_volumes.copy().loc[sell_binom.index]
    
    # Split up class_forecasts dataframe into bid/supply price class forecasts
    purchase_classes_columns = list(map(lambda string: float(string[:-1]), class_forecasts.filter(regex="P").columns))
    purchase_classes_columns.sort(reverse=True)
    sell_classes_columns = list(map(lambda string: float(string[:-1]), class_forecasts.filter(regex="S").columns))
    sell_classes_columns.sort(reverse=False)
    
    # Prepare output dataframe
    bid_volume_proportions = pd.DataFrame()
    
    for datetime in class_forecasts.index:
        purchase_proportions = purchase_active_bids.copy()
        for count, purchase_class in enumerate(purchase_classes_columns):
            if count == 0:
                purchase_proportions.loc[purchase_class] = class_forecasts.loc[datetime, f"{purchase_class}P"]
            else:
                class_interval = (purchase_proportions.index >= round(purchase_class,1)) & (purchase_proportions.index < round(purchase_classes_columns[count-1],1))
                class_bids = purchase_proportions.loc[class_interval]
                if class_bids.sum().sum() == 0:
                    purchase_proportions.loc[round(purchase_class,1)] = class_forecasts.loc[datetime, f"{purchase_class}P"]
                else:
                    purchase_proportions.loc[class_interval] /= class_bids.sum() / class_forecasts.loc[datetime, f"{purchase_class}P"]
        purchase_proportions = purchase_proportions.sort_index(ascending=False)
        
        
        sell_proportions = sell_active_bids.copy()
        for count, sell_class in enumerate(sell_classes_columns):
            if count == 0:
                sell_proportions.loc[sell_class] = class_forecasts.loc[datetime, f"{sell_class}S"]
            else:
                class_interval = (sell_proportions.index <= round(sell_class,1)) & (sell_proportions.index > round(sell_classes_columns[count-1],1))
                class_bids = sell_proportions.loc[class_interval]
                if class_bids.sum().sum() == 0:
                    sell_proportions.loc[round(sell_class,1)] = class_forecasts.loc[datetime, f"{sell_class}S"]
                else:
                    sell_proportions.loc[class_interval] /= class_bids.sum() / class_forecasts.loc[datetime, f"{sell_class}S"]
        sell_proportions = sell_proportions.sort_index(ascending=True)
        
        if datetime == class_forecasts.index[0]:
            output_columns = list(map(lambda i: str(i)+"P", purchase_proportions.index)) + list(map(lambda i: str(i)+"S", sell_proportions.index))
            bid_volume_proportions = pd.DataFrame(index=class_forecasts.index, columns=output_columns)
    
        bid_volume_proportions.loc[datetime] = np.concatenate((purchase_proportions.values.flatten(), sell_proportions.values.flatten()))
        
    return(bid_volume_proportions)


class xmodel():
    def __init__(self, lag_structure, verbose, train_params):
        self.lag_structure = lag_structure
        self.verbose = verbose
        self.train_params = train_params
   
    def ingest_data(self, train_prices, train_bm, train_planned, params_tuple):
        self.price_classes, self.xm_predictors = params_tuple
        self.train_xm_predictors = self.xm_predictors.iloc[:-1,:]
        self.test_xm_predictors = self.xm_predictors.iloc[-1,:].to_frame().T

    def train(self):
        pass
    
    def forecast(self):
        forecast_start_date = dt.datetime.combine(self.test_xm_predictors.index[0], dt.datetime.min.time())
        forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_start_date+dt.timedelta(hours=23), freq='H')
        self.forecasts_df = pd.DataFrame(columns=self.price_classes.columns, index=forecast_dates)

        if not hasattr(self, "predictor_column_sets"):
            self.predictor_column_sets = pd.DataFrame(columns=self.price_classes.columns, index=list(range(24)))
            
            # Train and forecast values for each given price class at each given hour.
            for target_hour in range(24):
                for target_class in self.price_classes.columns:
                    # Fetch the time series corresponding to the price target_class at the target_hour
                    target = self.price_classes.loc[self.price_classes.index.hour==target_hour, target_class]
                    target.index = target.index.date

                    # These dataframes will store all the 'predictors' of the target class, consisting of lagged values of other price
                    # classes and of the exogenous classes (wind forecast, total volume, electricity price, etc.).
                    #
                    # train_predictors: predictor values used for training
                    # forecast_predictors: predictor values used for forecasting day-ahead prices
                    #
                    predictors_columns = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

                    # Build the predictor dataframes (from price classes)
                    for price_class in self.price_classes.columns:
                        for predictor_hour in range(24):
                            # Get the specific dependency of the target price class on the current price_class at the current hour
                            lags = get_lags(self.lag_structure, target_class, target_hour, price_class, predictor_hour)

                            # For each given lag, populate train_predictors and forecast_predictors with the appropriate columns/values
                            for lag in lags:
                                predictors_columns.append(f"{price_class}_H{predictor_hour}-{lag}")

                    # (Continued) Build the predictor dataframes (from exogenous, non-planned classes)
                    for exog_class in self.exog_classes_columns:
                        for predictor_hour in range(24):
                            # Get the specific dependency of the target price class on the current price_class at the current hour
                            lags = get_lags(self.lag_structure, target_class, target_hour, exog_class, predictor_hour)

                            # For each given lag, populate train_predictors and forecast_predictors with the appropriate columns/values
                            for lag in lags:
                                predictors_columns.append(f"{exog_class}_H{predictor_hour}-{lag}")

                    # (Continued) Build the predictor dataframes (from exogenous, planned classes)
                    for planned_class in self.planned_classes_columns:
                        for predictor_hour in range(24):
                            # Get the specific dependency of the target price class on the current price_class at the current hour
                            lags = get_lags(self.lag_structure, target_class, target_hour, planned_class, predictor_hour)
                            lags = list(map(lambda lag: lag-1, lags))

                            # For each given lag, populate train_predictors and forecast_predictors with the appropriate columns/values
                            # Note that with planned forecasts, we know the (forecasted) value for the delivery hour that we are 
                            # forecasting, so we let the target variable depend on this, i.e. include the unlagged version as predictor
                            for lag in lags:
                                predictors_columns.append(f"{planned_class}_H{predictor_hour}-{lag}")

                    self.predictor_column_sets.loc[target_hour, target_class] = predictors_columns

        for target_hour in range(24):
            for target_class in self.price_classes.columns:
                if self.verbose: print(f"Now on price class {target_class} at hour {target_hour}.")
                
                target = self.price_classes.loc[self.price_classes.index.hour==target_hour, target_class]
                target.index = target.index.date
                predictors_columns = self.predictor_column_sets.loc[target_hour, target_class]

                # Match the column arrangement of train_predictors with forecast_predictors.
                # This is an important safety measure to ensure that the training and forecasting processes are consistent with each other.
                train_predictors = self.train_xm_predictors[predictors_columns]
                test_predictors = self.test_xm_predictors[predictors_columns]

                # Remove rows (samples) with NAs (that arose due to the lagging process in the loop above)
                # so that none of our samples have NA values.
                notna_train_predictors_loc = train_predictors.notna().all(axis=1)
                train_predictors = train_predictors.loc[notna_train_predictors_loc]
                target = target.loc[notna_train_predictors_loc]

                ## Fit the LASSO model (with optimal lambda found via cross-validation)
                # Generate forecast.
                #
                # Get scaling parameters (mean and variance/standard deviation) of train_predictors and target for the LASSO fitting.
                # This is important for allowing interpretability of the model coefficients and to help the algorithm with convergence.
                # The X-model paper implements this same procedure.
                predictors_scale = prep.StandardScaler().fit(X=train_predictors)
                target_scale = prep.StandardScaler().fit(X=pd.DataFrame(target))

                # Apply scaling
                train_predictors_scaled = predictors_scale.transform(X=train_predictors)
                target_scaled = target_scale.transform(pd.DataFrame(target))

                # Before passing into the predict() function, we make sure to scale forecast_predictors
                # using the same scaling parameters that were applied to training_predictors. We then scale back the forecast
                # using the same scaling parameters that were applied to target (but in reverse order).
                model_fit = linear_model.LassoCV(**self.train_params).fit(X=train_predictors_scaled, y=target_scaled.flatten())
                test_predictors_scaled = predictors_scale.transform(test_predictors)
                scaled_forecast = model_fit.predict(test_predictors_scaled)
                forecast = target_scale.inverse_transform(scaled_forecast)

                self.forecasts_df.loc[self.forecasts_df.index.hour==target_hour, target_class] = forecast
        
        # Reconstruct price classes into individual price forecasts (skipping the bid reconstruction step altogether)
        final_forecast = forecast_prices_from_classes(self.forecasts_df)
        
        return(final_forecast)