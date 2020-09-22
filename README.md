# Electricity Price Forecasting:
### Applying Statistical and Machine Learning Methods to the Irish Day-Ahead Electricity Market in a Supervised Learning Framework

## Abstract:
Accurate electricity price forecasting has become a substantial requirement since the liberalisation of the electricity markets, which has created a more competitive system of electricity generation and distribution. For example, electricity prices in the day-ahead market (DAM) are considered by energy traders when making decisions. However, electricity prices display characteristics that are difficult from a modelling perspective, including heteroscedasticity (non-constant variance), sharp price spikes and several levels of seasonality.

In this paper, we present some of the methodologies in the current wealth of literature on electricity price forecasting, with the aim of forecasting electricity spot prices in the Irish DAM. More specifically, we use methods in time series and machine learning to forecast prices for all 24 hours of each DAM auction operated by the Single Electricity Market Operator (SEMO).

First, we give a brief overview of the electricity market and its operation as far as it relates to our goal of price forecasting in the DAM. Secondly, some important information is given about the data used for this project. Thirdly, we give a short explanation of each model that was used and then provide details on how each one was structured and prepared for the modelling task. Then, most importantly, we present results on the performance of each model relative to a benchmark and a brief note on the importance of predictors. Finally, we give conclusions based on the results and note some areas for further development.

## GitHub Description:
This is the repository for the code, datasets, etc. created for my MSc dissertation on electricity price forecasting. The paper is found in Electricity_Price_Forecasting.pdf. The file "Method Evaluation.ipynb" contains the codes used to run the walk-forward validation (as outlined in Section 4.1 and defined in packages\\tools.py) for all the models (outlined in Section 3 and defined in packages\\models.py).

The Datasets directory contains all datasets used in the project. The Figures directory contains all plots and tables used in Electricity_Price_Forecasting.pdf. The Results directory contains the walk-forward validation results for analysis. The Variables directory is an optional directory contianing the large files (X-Model) that would take several minutes to build from scratch.

## Directory Tree:
```
│   Electricity_Price_Forecasting.pdf
│   Method Evaluation.ipynb
│   README.md
│   Walk-Forward Results.ipynb
│
├───Datasets
│   │   BidAskCurvve1.csv
│   │   BidAskCurvve2.csv
│   │   BidAskCurvve3.csv
│   │   BidAskCurvve4.csv
│   │   BMInfo1.csv
│   │   BMInfo2.csv
│   │   BMInfo3.csv
│   │   DAMPrices.csv
│   │   DAMPrices2.csv
│   │   DemandForecast.csv
│   │   DemandForecast2.csv
│   │   SEMO Data Publication Guide Issue 3.1.pdf
│   │   SEMOpx Data Publication Guide Issue 6.0.pdf
│   │   WindForecast.csv
│   │   WindForecast2.csv
│   │
│   └───Writeup
│           _jobs.csv
│
├───Figures
│   ├───Data
│   │       full_prices.png
│   │       other_datasets.png
│   │       prices_2020_5_23.PNG
│   │       SEMOpx_Data_Publication_Guide.zip
│   │       summer19_prices.png
│   │       summer20_prices.png
│   │       winter18_prices.png
│   │       winter19_prices.png
│   │
│   ├───Neural Networks
│   │       ffnn.PNG
│   │       linear.png
│   │       LSTM3-C-line.png
│   │       LSTM3-chain.png
│   │       LSTM3-focus-C.png
│   │       LSTM3-focus-f.png
│   │       LSTM3-focus-i.png
│   │       LSTM3-focus-o.png
│   │       LSTM3-SimpleRNN.png
│   │       relu.png
│   │       RNN-longtermdependencies.png
│   │       RNN-rolled.png
│   │       RNN-shorttermdepdencies.png
│   │       RNN-unrolled.png
│   │       sigmoid.png
│   │       tanh.png
│   │
│   ├───Random Forests
│   │       boostrap.PNG
│   │       rf_decision_tree.JPG
│   │
│   ├───Results
│   │       mae_entire_period_hourly_plot.PNG
│   │       mae_entire_period_overall_plot.png
│   │       mae_entire_period_overall_table.PNG
│   │       negative_price_spike_distribution.png
│   │       positive_price_spike_distribution.png
│   │       price_spike_interval.png
│   │       rf_hyperparameter.png
│   │       rmse_entire_period_hourly_plot.PNG
│   │       rmse_entire_period_overall_plot.png
│   │       rmse_entire_period_overall_table.PNG
│   │       rmse_midcovid_hourly_plot.png
│   │       rmse_precovid_hourly_plot.png
│   │       rmse_scaled_covid_overall_plot.png
│   │       rmse_scaled_price_spikes_plot.png
│   │       rmse_unscaled_covid_overall_plot.png
│   │       rmse_unscaled_price_spikes_plot.png
│   │       sarimax_hyperparameter.png
│   │       sigmoid_ffnn_hyperparameter.png
│   │       variable_importances_plot.png
│   │       variable_importances_table.png
│   │
│   ├───Time Series
│   │       1h - Electricity Prices.png
│   │       1h_24h - Electricity Prices (zoomed).png
│   │       1h_24h - Electricity Prices.png
│   │       acf1.png
│   │       ar2.png
│   │       ar2_acf_pacf.png
│   │       arma23.png
│   │       arma23_acf_pacf.png
│   │       diff1_diff24_prices_acf_pacf.png
│   │       diff1_diff24_prices_acf_pacf_zoomed.png
│   │       diff1_prices_acf_pacf.png
│   │       diff24_prices_acf_pacf.png
│   │       full_prices_acf_pacf.png
│   │       increasing_mean.png
│   │       increasing_variance.png
│   │       ma3.png
│   │       ma3_acf_pacf.png
│   │       may2020_diff.png
│   │       may2020_diff24.png
│   │       may2020_orig.png
│   │       non_constant_mean.png
│   │       Original Electricity Prices.png
│   │       pacf1.png
│   │       stationary.png
│   │
│   └───X-Model
│           legend1.png
│           legend2.png
│           legend_mean_demand.png
│           legend_mean_supply.png
│           mean_demand_classes.png
│           mean_demand_classes_zoomed.png
│           mean_demand_volumes.png
│           mean_supply_classes.png
│           mean_supply_classes_zoomed.png
│           mean_supply_volumes.png
│           supply_demand_curves1.png
│           supply_demand_curves2.png
│           x_model_dependency.PNG
│
├───packages
│   	models.py
│   	tools.py
│
├───Results
│       arx_forecasts.csv
│       ar_forecasts.csv
│       ffnn_hyperparameter_errors.csv
│       naive_1d_forecasts.csv
│       naive_1y_forecasts.csv
│       naive_7d_forecasts.csv
│       relu_ffnn_forecasts.csv
│       relu_rnn_forecasts.csv
│       rf_forecasts.csv
│       rf_hyperparameter_errors.csv
│       rf_vi.csv
│       sarimax_forecasts.csv
│       sarimax_hyperparameter_errors.csv
│       sarima_forecasts.csv
│       sigmoid_ffnn_forecasts.csv
│       sigmoid_rnn_forecasts.csv
│       tanh_ffnn_forecasts.csv
│       tanh_rnn_forecasts.csv
│       xmodel_forecasts_april.csv
│       xmodel_forecasts_february.csv
│       xmodel_forecasts_january.csv
│       xmodel_forecasts_july.csv
│       xmodel_forecasts_june.csv
│       xmodel_forecasts_march.csv
│       xmodel_forecasts_may.csv
│
└───Variables
        .gitattributes
        price_classes.csv
        xm_predictors.csv
```
