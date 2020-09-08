import pandas as pd
import datetime as dt
import time
import re
import warnings

warnings.filterwarnings('ignore')

from functools import reduce

from sklearn import metrics

from packages import models

###########################################################################################################################################
# Balancing Market data

def read_bm_price_data(filenames=['Datasets/BMInfo1.csv', 'Datasets/BMInfo2.csv', 'Datasets/BMInfo3.csv'], columns=['StartTime', 'ImbalancePrice'], date_format='%Y-%m-%d %H:%M:%S'):
    """
    Reads in balancing market prices from Datasets directory.

    Parameters
    ----------
        filenames : list of str
            List of filenames containing balancing market price data.
        columns : list of str
            List of columns to read from the files.
        date_format : str, default='%Y-%m-%d %H:%M:%S'
            Date format to be parsed from string to datetime for the date columns in the dataset.

    Returns
    -------
        bm_data : pandas.DataFrame
            Pre-processed balancing market price data.
    """
    # Create date parser
    date_parse = lambda date: dt.datetime.strptime(date, date_format) + dt.timedelta(hours=1)
    
    # Read in dataset/s
    bm_data = [pd.read_csv(filenames[i], usecols=columns, parse_dates=True, index_col='StartTime', date_parser=date_parse) for i in range(len(filenames))]
    
    # Combine dataset/s (and filter out duplicates)
    bm_data = reduce(lambda df1, df2: df1.append(df2), bm_data)
    bm_data.freq = 'H'
    
    bm_data = bm_data.loc[[not val for val in bm_data.index.duplicated()]]
    
    # Remove last day of data if last delivery hour is not 23:00
    if bm_data.index.hour[-1] != 23:
        last_date = dt.datetime.combine(bm_data.index.date[-1], dt.datetime.min.time()) - dt.timedelta(hours=1)
        bm_data = bm_data.loc[:last_date]
    
    # Get only hourly data (since BM data has half-hourly granularity)
    bm_data = bm_data.loc[bm_data.index.minute == 0]
    
    # Get list of instances where the index is duplicated
    count_df = bm_data.groupby(bm_data.index).count()
    duplicates = count_df.loc[count_df['ImbalancePrice'] != 1].index
    
    # Remove rows with duplicate indices (by replacing all the rows with their mean)    
    for date in duplicates:
        duplicates_df = bm_data.loc[bm_data.index == date]
        average = duplicates_df.mean().values
        bm_data.drop(date, inplace=True)
        bm_data = bm_data.append(pd.DataFrame(average, index=[date], columns=['ImbalancePrice']))
    
    # Get list of dates that do not have exactly 24 data points (hours)
    count_df = bm_data.groupby(bm_data.index.date).count()
    missing_values = count_df.loc[count_df['ImbalancePrice'] != 24]

    # Clean data - Insert missing values (obtained above) with the overall mean
    for date in missing_values.index:
        date_hours = bm_data.loc[bm_data.index.date == date].index.hour

        for hour in range(24):
            if hour not in date_hours:
                missing_value_loc = dt.datetime.combine(date, dt.datetime.min.time()) + dt.timedelta(hours=hour)
                imputed_value = pd.DataFrame(columns=['ImbalancePrice'], index=[missing_value_loc])
                bm_data = bm_data.append(imputed_value)
    
    bm_data.fillna(bm_data.mean(), inplace=True)
    bm_data.sort_index(inplace=True)
    
    # Relabel the index
    bm_data.index.name = 'DeliveryPeriod'

    return(bm_data)

###########################################################################################################################################
# Demand/Wind forecast data

def read_forecast_data(forecast_type, filename=None, forecast_column=['StartTime', 'AggregatedForecast'], date_format='%Y-%m-%d %H:%M:%S'):
    """
    Reads in forecast demand or wind load from Datasets directory.

    Parameters
    ----------
        forecast_type : string in ['Wind', 'Forecast']
        filename : list of str
            List of filenames containing the forecast data.
        forecast_column : list of str
            List of columns to read from the files.
        date_format : str, default='%Y-%m-%d %H:%M:%S'
            Date format to be parsed from string to datetime for the date columns in the dataset.

    Returns
    -------
        forecast_data : pandas.DataFrame
            Pre-processed forecast data.
    """
    # Set the filenames to be read in when forecast_type is specified.
    if filename is None:
        if forecast_type == 'Wind':
            filename = ['Datasets/WindForecast.csv', 'Datasets/WindForecast2.csv']
        elif forecast_type == 'Demand':
            filename = ['Datasets/DemandForecast.csv', 'Datasets/DemandForecast2.csv']
    
    # Create date parser
    date_parse = lambda date: dt.datetime.strptime(date, date_format) + dt.timedelta(hours=1)
    
    # Read in dataset/s
    forecast_data1 = pd.read_csv(filename[0], usecols=forecast_column, parse_dates=True, index_col='StartTime', date_parser=date_parse)
    forecast_data2 = pd.read_csv(filename[1], usecols=forecast_column, parse_dates=True, index_col='StartTime', date_parser=date_parse)
    
    # Combine dataset/s
    forecast_data = forecast_data1.append(forecast_data2)
    
    # Remove duplicates
    forecast_data = forecast_data.loc[[not val for val in forecast_data.index.duplicated()]]
    
    # Rename index and column
    forecast_data.index.name = 'DeliveryPeriod'
    forecast_data.columns = [forecast_type]
    
    return(forecast_data)

###########################################################################################################################################
# Electricity price data

def read_price_data(filenames=['Datasets/DAMPrices.csv', 'Datasets/DAMPrices2.csv'], columns=['DeliveryPeriod', 'EURPrices'], date_format='%Y-%m-%d %H:%M:%S'):
    """
    Reads in day-ahead market prices from Datasets directory.

    Parameters
    ----------
        filenames : list of str
            List of filenames containing day-ahead market price data.
        columns : list of str
            List of columns to read from the files.
        date_format : str, default='%Y-%m-%d %H:%M:%S'
            Date format to be parsed from string to datetime for the date columns in the dataset.

    Returns
    -------
        price_data : pandas.DataFrame
            Pre-processed DAM price data.
    """
    # Create date parser
    date_parse = lambda date: dt.datetime.strptime(date, date_format) + dt.timedelta(hours=1)
    
    # Read in datasets
    price_data1 = pd.read_csv(filenames[0], usecols=columns, parse_dates=True, index_col='DeliveryPeriod', date_parser=date_parse)
    
    price_data2 = pd.read_csv(filenames[1], usecols=columns)
    price_data2.index = pd.to_datetime(price_data2['DeliveryPeriod'], format='%d/%m/%Y %H:%M') + dt.timedelta(hours=1)
    price_data2.drop('DeliveryPeriod', axis=1, inplace=True)
    
    # Combine datasets
    price_data = price_data1.append(price_data2)
    price_data.freq = 'H'
    
    # Do DST adjustment - 23-hour days have their missing hour replaced with the average of the two days before and after the missing hour.
    #                     25-hour days have the two same hours replaced by their average.
    price_data = price_dst_adjustment(price_dst_adjustment(price_data))

    return(price_data)


def find_dst_index(time_step_id_dataframe, number_of_hours):
    """
    Given a set of datetime.hour values for a given date (freq='H'), return the index/hour that is either missing (if
    number_of_hours==23) or appears twice (if number_of_hours==25).

    Parameters
    ----------
        time_step_id_dataframe : pandas.core.indexes.numeric.Int64Index
        number_of_hours : int

    Returns
    -------
        time_step_id : int 
            The missing or duplicated hour for the given DST day.
    """
    # If list is missing an hour, return that missing hour
    if number_of_hours == 23:
        for i, time_step_id in enumerate(time_step_id_dataframe):
            if i < time_step_id:
                return(i+1)
            elif i == number_of_hours-1:
                return(23)
                
    # If list has two duplicate hours, return the duplicated hour
    elif number_of_hours == 25:
        for j, time_step_id in enumerate(time_step_id_dataframe):
            if j+1 > time_step_id:
                return(time_step_id)


def price_dst_adjustment(df):
    """ 
    Given a dataframe (of electricity prices), make DST adjustments so that days with 23 or 25 hours
    are imputed/reduced to 24 hours using simple averaging.

    Parameters
    ----------
        df : pandas.DataFrame
            Unadjusted electricity prices dataframe.

    Returns
    -------
        df : pandas.DataFrame
            Adjusted electricity prices dataframe.
    """
    # Fetch dataframe of dates that do not have exactly 24 hours of data.
    df_count = df.groupby([df.index.date]).count()
    dst_dates = df_count.loc[(df_count['EURPrices']) != 24,:]

    # If there are no such dates (i.e. df has already been adjusted).
    if dst_dates.shape[0] == 0:
        return(df)

    for i in range(dst_dates.shape[0]):
        # Fetch the number of hours of data for the given DST date. This should (only) be either 23 or 25.
        dst_date = dst_dates.index[i]
        number_of_hours = dst_dates.iloc[i,0]

        # Get the price data for the specific dst_date
        df_dst_data = df.loc[df.index.date == dst_date]
        
        # Find the specific hour that's either duplicated (for 25-hour days) or missing (for 23-hour days).
        dst_index = find_dst_index(df_dst_data.index.hour, number_of_hours)
        
        # If 23-hour day, get the average of the price data for the adjacent hours.
        if number_of_hours == 23:
            # Fetch adjacent prices, e.g. if missing prices is for 3rd hour, then we fetch prices for 2nd and 4th hour.
            previous_price = df_dst_data.loc[df_dst_data.index.hour == dst_index-1]
            next_day_price = df.loc[df.index.date == dst_date+dt.timedelta(days=1)]
            next_price = next_day_price.loc[next_day_price.index.hour == 0]
            adjacent_prices = previous_price.append(next_price)
            
            # Calculate the average of the two hours of price data
            average_values = adjacent_prices.mean(axis=0).values[0]
            
            # Insert this new price into the original dataframe containing reduced price data.
            new_price = pd.DataFrame(dict(EURPrices=average_values), index=[dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            df = df.append(new_price)
            
            # Clean up (reset index)
            df.index = pd.to_datetime(df.index)
            df.index.name = 'DeliveryPeriod'
            
        # If 25-hour day, replace these with the average price of the two duplicate hours.
        elif number_of_hours == 25:
            # Fetch duplicate prices
            duplicate_prices = df_dst_data.loc[df_dst_data.index.hour == dst_index]

            # Calculate the average of the two hours of price data
            average_values = duplicate_prices.mean(axis=0).values[0]
            
            # Delete the two rows of duplicate hours
            df.drop(duplicate_prices.index, inplace=True)
            
            # Insert this new price into the original dataframe containing reduced price data.
            new_price = pd.DataFrame(dict(EURPrices=average_values), index=[dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            df = df.append(new_price)
            df.index = pd.to_datetime(df.index)

    # Re-order axis by datetime (ascending order) and remove duplicates.
    df.sort_index(axis=0, inplace=True)
    df = df.loc[[not val for val in df.index.duplicated()]]
    
    return(df)

###########################################################################################################################################
# Bid Curve data

def read_curve_data(filenames=['Datasets/BidAskCurvve1.csv','Datasets/BidAskCurvve2.csv','Datasets/BidAskCurvve3.csv','Datasets/BidAskCurvve4.csv'], columns=['DeliveryDay', 'TimeStepID', 'PurchaseVolume', 'PurchasePrice', 'SellVolume', 'SellPrice']):
    """
    Reads in DAM bid curve data.

    Parameters
    ----------
        filenames : list of str
            List of filenames containing day-ahead market bid curve data
        columns : list of str
             List of columns to read from the files.

    Returns
    -------
        ba : pandas.DataFrame
            Dataframe containing demand and supply curve data for the day-ahead market auctions (with some basic pre-processing).
    """
    # Read in dataframes into a list
    ba = [pd.read_csv(filenames[i], usecols=columns) for i in range(len(filenames))]
    
    # Combine list of dataframes into a single dataframe
    ba = reduce(lambda df1, df2: df1.append(df2), ba)
    
    # Data cleaning (ensuring consistent data types, remove duplicate days/hours, sort by delivery day/time, etc.)
    ba = clean_ba_data(ba).drop_duplicates()
    
    return(ba)


def clean_TimeStepID(TimeStepID):
    """
    Ancillary function for cleaning bid curve data.
    """
    # If no need to edit, return the same value
    if type(TimeStepID) == int:
        return(TimeStepID)
    
    # If TimeStepID has a letter.
    if re.search('.*B$', TimeStepID):
        pos = re.search('B', TimeStepID).start()
    else:
        return(int(TimeStepID))
    
    # Remove letter and convert to int
    return(int(TimeStepID[:pos]))


def clean_ba_data(dataframe):
    """
    Function to clean bid curve data frame
    """
    # Ensure data types are consistent, i.e. DeliveryDay is date, TimeStepID is int, etc.
    dataframe.loc[:,'DeliveryDay'] = pd.to_datetime(dataframe['DeliveryDay'], format='%Y-%m-%d')
    dataframe.loc[:,'TimeStepID'] = dataframe['TimeStepID'].apply(clean_TimeStepID)
    
    # Convert TimeStepID to zero-index, i.e. {0,...,23}
    dataframe.loc[:,'TimeStepID'] = dataframe.loc[:,'TimeStepID']-1

    # Combine DeliveryDay and TimeStepID into one date column, 'DeliveryPeriod', and set as index
    dataframe['DeliveryPeriod'] = [day + dt.timedelta(hours=hour) for (day, hour) in zip(dataframe['DeliveryDay'], dataframe['TimeStepID'])]
    dataframe.set_index('DeliveryPeriod', inplace=True)
    dataframe = dataframe.sort_values(by=['DeliveryPeriod'])
    
    # Remove redundant columns
    dataframe.drop(['DeliveryDay', 'TimeStepID'], axis=1, inplace=True)
    
    return(dataframe)

###########################################################################################################################################
# Walk-forward evaluation of a forecasting model

def walk_forward_evaluation(model, price_data, bm_data=None, planned_data=None, bid_curve_data=None, starting_window_size=None, moving_window=False, start=None, end=None, logs=True):
    """
    Evalutes a forecasting model using the given price_data and (optional) balancing market data,
    wind/demand forecast data, and bid curve data. The general procedure is as follows:
        
        * Create initial training data window. We only use the data that would be available the day before
          the first forecast date given by the start parameter.
        * For each forecast_date in dates_between(start, end):
            - Ingest data. The data required by the model is reformatted (as needed) into suitable train/test input
              and stored in the model object for later training/forecasting.
            - Train. The model is trained to the available data prior to forecast_date.
            - Forecast. Forecast is generated for forecast_date.
            - Store forecasts in a dataframe.
        * Calculate RMSE and MAE for the whole period.
        * Return model object, model forecasts, RMSE and MAE.

    The above procedure applies for the following model classes: naive(), random_forest(), ARX(), SARIMAX(), ffnn(), rnn().

    In the case of xmodel(), the majority of the preprocessing step is done before the loop
    in order to adapt the model to the ingest->train->forecast procedure.

    Parameters
    ----------
        model : class
            Class object defining the forecasting model, with class methods
            self.ingest_data(), self.train() and self.forecast().
        price_data : pandas.DataFrame
            Electricity price data.
        bm_data : pandas.DataFrame
            Balancing market price data.
        planned_data : pandas.DataFrame
            Wind and demand forecast data.
        bid_curve_data : pandas.DataFrame
            DAM supply and demand curve data.
        starting_window_size : int
            number of days of data to start the training set on, i.e. training set size
            for first forecast iteration on start date.
        moving_window : bool
            To specify whether the training window is a moving window (True) or an expanding window (False).
        start : datetime.datetime
            Date on which to start the walk-forward validation.
        end : datetime.datetime
            Date to end the walk-forward validation on (inclusive).
        logs : bool, default=True
            Specifies whether to print overall execution time of the walk-forward validation
            for the entire start-to-end period. True prints out the logs.

    Returns
    -------
        model : class
            The model object after it has been modified by the walk_forward_evaluation() function.
        forecasts_df : pandas.DataFrame
            DataFrame of original prices and corresponding model forecasts
        rmse : int
            Root-mean-square error (RMSE) for the entire period represented by forecasts_df
        mae : int
            Mean absolute error (MAE) for the entire period represented by forecasts_df
"""
    if logs:
        start_time = time.time()
    
    # Validation to ensure start date <= end date.
    if start is not None and end is not None:
        if start > end:
            raise Exception(f"Cannot have start date after end date.")
    
    # Validation to ensure the number of days of training data is at least starting_window_size.
    if starting_window_size is not None:
        if (start-price_data.index[0]).days < starting_window_size:
            raise Exception(f"Not enough data for training: starting_window_size={starting_window_size}, train_size={start-data.index[0]}")
    
    # Fetch datetime index for initial training data window
    if starting_window_size is None:
        train_dates = list(pd.date_range(start=price_data.index[0], end=start-dt.timedelta(hours=1), freq='H'))
    else:
        train_dates = list(pd.date_range(end=start-dt.timedelta(hours=1), periods=24*starting_window_size, freq='H'))

    # Get initial price data
    train_price_data = price_data.loc[train_dates,:]
    
    # Get initial balancing market data
    train_bm_data = None if bm_data is None else bm_data.loc[train_dates,:]
    
    # Fetch datetime index for initial planned (wind & forecast) data
    if planned_data is not None:
        if starting_window_size is None:
            train_planned_dates = pd.date_range(start=planned_data.index[0], end=start+dt.timedelta(hours=23), freq='H')
        else:
            train_planned_dates = pd.date_range(end=start+dt.timedelta(hours=23), periods=24*(starting_window_size+1), freq='H')
    
    # Get initial planned (wind & demand) data
    train_planned_data = None if planned_data is None else planned_data.loc[train_planned_dates,:]
    
    # X-Model data preprocessing
    if bid_curve_data is not None:
        try:
            print("Reading price class data and xm_predictors.")
            # Read in prebuilt dataframe of predictors and price classes.
            xm_predictors = pd.read_csv('Variables/xm_predictors.csv', index_col='DeliveryPeriod', parse_dates=True)
            xm_predictors.index = xm_predictors.index.date
            price_classes = pd.read_csv('Variables/price_classes.csv', index_col='DeliveryPeriod', parse_dates=True)
            
            # Get initial window of X-Model predictors and price classes.
            train_price_classes = price_classes.loc[price_classes.index >= train_dates[0]]
            train_price_classes = train_price_classes[train_price_classes.index <= train_dates[-1]]
            train_xm_predictors = xm_predictors.loc[xm_predictors.index >= train_dates[0]]
            train_xm_predictors = train_xm_predictors.loc[train_xm_predictors.index <= (train_dates[-1] + dt.timedelta(days=1))]

            # Combine DAM prices and BM prices into one dataframe.
            exog_classes = reduce(lambda df1, df2: df1.join(df2), [price_data, bm_data])
            planned_classes = planned_data.copy()
            
            # Store variable names in model object
            model.exog_classes_columns = exog_classes.columns
            model.planned_classes_columns = planned_classes.columns
            
            # Match start (and end) dates of DAM price data, BM data, planned data and bid curve data.
            latest_start_date = max([df.index[0] for df in [price_data, bm_data, planned_data, bid_curve_data]])

            last_day_of_data = dt.datetime.combine(planned_data.index.date[-1], dt.datetime.min.time())
            test_data_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq='H')
            
            train_price_classes = price_classes.loc[price_classes.index >= train_dates[0]]
            train_price_classes = train_price_classes[train_price_classes.index <= train_dates[-1]]
            train_xm_predictors = xm_predictors.loc[xm_predictors.index >= train_dates[0]]
            train_xm_predictors = train_xm_predictors.loc[train_xm_predictors.index <= (train_dates[-1] + dt.timedelta(days=1))]

        except:
            # If the predictors have not been prebuilt and stored locally.
            print("Could not find xm_predictors and/or price_classes locally. Creating them first...")
            train_bid_curve_data = bid_curve_data.loc[bid_curve_data.index >= train_dates[0]]
            train_bid_curve_data = train_bid_curve_data.loc[train_bid_curve_data.index <= train_dates[-1]]

            # Calculate mean volumes per price point
            print("Calculating price classes from initial data")
            mean_purchase, mean_sell = models.get_mean_volumes(train_bid_curve_data)

            # Generate 'mean' supply and demand curves
            mean_purchase_curve = models.to_price_curve(mean_purchase, curve_type='purchase')
            mean_sell_curve = models.to_price_curve(mean_sell, curve_type='sell')

            # Create price classes from mean supply and demand curves
            purchase_classes, sell_classes = models.create_price_classes(mean_purchase_curve, mean_sell_curve, class_length=35000)

            # Reduce bid data to bid volumes for each price class
            print("Reducing data to price class bid volumes.")
            purchase_reduced, sell_reduced = models.reduce_bid_ask_data(bid_curve_data, purchase_classes, sell_classes)

            # Apply DST adjustments
            print("Applying DST adjutsments to bid curve data.")
            purchase = models.dst_adjustment(purchase_reduced)
            sell = models.dst_adjustment(sell_reduced)

            # Data imputation for completely missing days (2019-11-22, 2020-02-12, 2020-04-19)
            purchase = models.clean_reduced_bid_data(purchase)
            sell = models.clean_reduced_bid_data(sell)
            
            # Combine DAM prices and BM prices into one df
            exog_classes = reduce(lambda df1, df2: df1.join(df2), [price_data, bm_data])
            planned_classes = planned_data.copy()

            # Match start (and end) dates
            latest_start_date = max([df.index[0] for df in [price_data, bm_data, planned_data, bid_curve_data]  ])

            last_day_of_data = dt.datetime.combine(planned_data.index.date[-1], dt.datetime.min.time())
            test_data_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq='H')

            train_purchase = purchase.loc[latest_start_date:last_day_of_data-dt.timedelta(hours=1)]
            train_sell = sell.loc[latest_start_date:last_day_of_data-dt.timedelta(hours=1)]
            train_exog_classes = exog_classes.loc[latest_start_date:last_day_of_data-dt.timedelta(hours=1)]
            train_planned_classes = planned_classes.loc[latest_start_date:test_data_index[-1]]

            # Build predictors
            print("Building predictors.")
            price_classes, xm_predictors = models.build_XM_predictors(train_purchase, train_sell,
                            train_exog_classes, train_planned_classes, model.lag_structure, verbose=True)

            # Get initial window of X-Model predictors and price classes.
            train_price_classes = price_classes.loc[price_classes.index >= train_dates[0]]
            train_price_classes = train_price_classes[train_price_classes.index <= train_dates[-1]]
            train_xm_predictors = xm_predictors.loc[xm_predictors.index >= train_dates[0]]
            train_xm_predictors = train_xm_predictors.loc[train_xm_predictors.index <= (train_dates[-1] + dt.timedelta(days=1))]

            # Store variable names in model object
            model.exog_classes_columns = exog_classes.columns
            model.planned_classes_columns = planned_classes.columns

            # Save predictors (and price classes) locally
            price_classes.to_csv('Variables/price_classes.csv')
            xm_predictors.to_csv('Variables/xm_predictors.csv')
        
    # Create dataframe to store errors
    forecast_index = pd.date_range(start=start, end=end+dt.timedelta(hours=23), freq='H')
    forecasts_df = pd.DataFrame(columns=['Forecast'], index=forecast_index)
    forecasts_df.insert(0, 'Original', price_data['EURPrices'].loc[forecast_index])
    
    # Loop through data to train and forecast iteratively over the expanding (or moving) window
    for i in range((end-start).days+1):
        # Ingest data
        if bid_curve_data is not None:
            model.ingest_data(train_price_data, train_bm_data, train_planned_data, (train_price_classes,train_xm_predictors))
        else:
            model.ingest_data(train_price_data, train_bm_data, train_planned_data, None)
        
        # Train model
        model.train()
        
        # Generate day-ahead forecast and store in forecasts_df dataframe
        forecast = model.forecast()
        forecasts_df.loc[forecast.index, 'Forecast'] = forecast.values
        
        # Drop last day of data if moving_window=True
        if moving_window:
            train_price_data.drop(train_price_data.index[:24], inplace=True)
            if bm_data is not None:
                train_bm_data.drop(train_bm_data.index[:24], inplace=True)
            if planned_data is not None:
                train_planned_data.drop(train_planned_data.index[:24], inplace=True)
            if bid_curve_data is not None:
                train_xm_predictors.drop(train_xm_predictors.index[:24], inplace=True)
                train_price_classes.drop(train_price_classes.index[:24], inplace=True)
        
        # Get datetime index for new date of data to be added to training data
        next_date = list(pd.date_range(start=train_price_data.index[-1]+dt.timedelta(hours=1), periods=24, freq='H'))
        
        # Fetch new DAM prices data and add to training data
        new_price_data = price_data.loc[next_date,:]
        train_price_data = train_price_data.append(new_price_data)
        
        print(f"Finished forecast for {forecast.index.date[0]}.")
        
        # Fetch new BM prices data and add to training data
        if bm_data is not None:
            new_bm_data = bm_data.loc[next_date,:]
            train_bm_data = train_bm_data.append(new_bm_data)
            
        # Fetch new forecast (wind & demand) data and add to training data
        if planned_data is not None:
            next_planned_date = list(pd.date_range(start=train_planned_data.index[-1]+dt.timedelta(hours=1), periods=24, freq='H'))
            new_planned_data = planned_data.loc[next_planned_date,:]
            train_planned_data = train_planned_data.append(new_planned_data)
            
        # Fetch new bid curve data and add to training data
        if bid_curve_data is not None:
            new_xm_predictors = xm_predictors.loc[(next_date[0]+dt.timedelta(days=1)).date(),:]
            train_xm_predictors = train_xm_predictors.append(new_xm_predictors)
            new_price_classes = price_classes.loc[next_date,:]
            train_price_classes = train_price_classes.append(new_price_classes)

    # Calculate RMSE and MAE for the entire period of the walk-forward evaluation
    rmse = metrics.mean_squared_error(forecasts_df['Original'], forecasts_df['Forecast'], squared=False)
    mae = metrics.mean_absolute_error(forecasts_df['Original'], forecasts_df['Forecast'])
    
    if logs:
        print(f"Execution time: {time.time()-start_time} seconds")
        
    return(model, forecasts_df, rmse, mae)


def get_resampled_errors(res_df, index_filter):
    """
    This takes a dataframe of forecasted (and original) hourly electricity price values
    and calculates rmse/mae values across different sampling rates.

    Parameters
    ----------
        res_df : pandas.DataFrame
            DataFrame of original and forecast prices.
        index_filter : string in ['date', 'dayofweek_and_hour', "'ayofweek', 'hour', 'month']
            Sampling rate.

    Returns
    -------
        errors : pandas.DataFrame
    """
    # Group dataframe values by index, with grouping determined by index_filter
    if index_filter == 'date':
        group = res_df.groupby(res_df.index.date)
    elif index_filter == 'dayofweek_and_hour':
        group = res_df.groupby([res_df.index.hour, res_df.index.dayofweek])
    elif index_filter == 'dayofweek':
        group = res_df.groupby(res_df.index.dayofweek)
    elif index_filter == 'hour':
        group = res_df.groupby(res_df.index.hour)
    elif index_filter == 'month':
        group = res_df.groupby(res_df.index.month)
        
    # Calculate RMSEs and MAEs for each group period
    rmses = group.apply(lambda df: metrics.mean_squared_error(df['Original'], df['Forecast'], squared=False))
    maes = group.apply(lambda df: metrics.mean_absolute_error(df['Original'], df['Forecast']))
    
    # Combine RMSEs and MAEs dataframes into one dataframe
    errors = pd.concat([rmses, maes], axis=1)
    errors.columns = ['RMSE', 'MAE']
    
    return(errors)


def walk_forward_loop(model_func, dates_to_forecast, model_params, lag_params, hyperparameter, prices, bm_prices, planned, logs=True):
    """
    A looping function to use the walk-forward validation on a set of non-adjacent (datetime) dates.
    This function is used for Section 4.4 - Hyperparameter Tuning.

    Parameters
    ----------
        model_func : class
            Class object defining the forecasting model, with class methods
            self.ingest_data(), self.train() and self.forecast().
        dates_to_forecast : list of datetime.datetime
            Dates to forecast on.
        model_params : dict
            Argument for model __init__() specifying model parameters
        lag_params : dict
            Argument for model __init__() specifying the data lags to be used as (exogenous) predictors
        hyperparameter : str
            Name of model hyperparameter to iteratively parameterise the model on.
            Note: the corresponding hyperparameter in the model_params must be a list of possible hyperparameter
            values to train the models on.
        prices : pandas.DataFrame
            Electricity price data.
        bm_prices : pandas.DataFrame
            Balancing market price data.
        planned : pandas.DataFrame
            Wind and demand forecast data.
        logs : bool, default=True
            Specifies whether to print overall execution time of the walk-forward validation
            for the entire start-to-end period. True prints out the logs.

    Returns
    -------
        errors : pandas.DataFrame
            Dataframe of overall RMSE and MAE of each hyperparameter value.
    """
    # Initialise dataframe to store errors (indexed by hyperparameter values given by model_params["init_params"][hyperparameter]).
    errors_index = model_params['init_params'][hyperparameter] if model_func == models.ffnn else model_params[hyperparameter]
    errors = pd.DataFrame(columns=['RMSE', 'MAE'], index=errors_index)
    errors.index.name = hyperparameter
    
    # Ensure that dates_to_forecast values are of type datetime.datetime
    dates_to_forecast = [dt.datetime.combine(date, dt.datetime.min.time()) if type(date) is dt.date else date for date in dates_to_forecast]

    # Fix hyperparameter value for walk-forward validation
    for param in errors.index:
        new_model_params = model_params.copy()
        
        # Create appropriate model_params dict as input for model __init__().
        if model_func == models.ffnn:
            new_model_params['init_params'][hyperparameter] = param
        else:
            new_model_params[hyperparameter] = param

        overall_res = pd.DataFrame(columns=['Original', 'Forecast'])
        
        # Run walk-forward validation for all dates in dates_to_forecast.
        for date in dates_to_forecast:
            # Initialise model
            model = model_func(model_params=new_model_params, lag_params=lag_params)
            
            # Run walk-forward validation function for the given date in dates_to_forecast,
            # and temporarily store the forecasts
            _, res, _, _ = walk_forward_evaluation(model, prices, bm_prices, planned, start=date, end=date, logs=False)
            overall_res = overall_res.append(res)
            
        # Calculate and store the RMSE and MAE for the forecasts over all the dates in dates_to_forecast
        rmse = metrics.mean_squared_error(overall_res['Original'], overall_res['Forecast'], squared=False)
        mae = metrics.mean_absolute_error(overall_res['Original'], overall_res['Forecast'])
            
        # Store the RMSE and MAE for the given hyperparameter.
        errors.loc[param, 'RMSE'] = rmse
        errors.loc[param, 'MAE'] = mae
        
        if logs: print(f"Finished for {hyperparameter}={param}")
    
    return(errors)