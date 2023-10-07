# Introduction

Freight demand forecasting is a crucial task for the trucking industry, as it enables decision-makers to allocate resources, optimize routes and vehicle usage, thereby directly contributing to cost savings and environmental sustainability. However, the lack of sufficient data points can pose a challenge to obtaining accurate forecasts using neural networks.

This paper proposes a data augmentation approach to compare forecasts using conventional ARIMA and SARIMA models for the regular time series and the augmented time series, as well as three types of neural networks: fully connected neural networks (FNNs), long short-term memory (LSTM) networks, and gated recurrent unit (GRU) encoder networks.

The data augmentation approach involves synthesizing new data points from the existing data set by adding random noise or by applying transformations such as shifting, scaling, and rotating the data. This helps to increase the size and diversity of the data set, which can improve the performance of neural network models.

The results of the experiments show that the data augmentation approach can significantly improve the accuracy of neural network forecasts, making them comparable to or even better than the forecasts of conventional time series models.

## Methodology

This section describes the methodology used to forecast freight demand using data augmentation.

### Data Preparation
Collect the historical freight demand data.

This is the first step in any forecasting process. It is important to collect as much data as possible, as this will lead to more accurate forecasts. The data can be collected from a variety of sources, such as industry reports, government agencies, and private companies. We use a dataset provided by efRouting, a smart truck dispatching company that provides diverse technological services.

### Clean and prepare the data

Outliers can skew the results of the forecasting process, so it is important to remove them from the data set. Missing values can also lead to inaccurate forecasts, so it is important to fill them in using a suitable method, such as linear interpolation or mean imputation. The data we used has been rigorously analyzed by the company in the past and is a viable time series. 

### Split the data into training and testing sets.

The training set is used to train the forecasting model, while the testing set is used to evaluate the performance of the model on unseen data. This helps to ensure that the model is able to generalize to new data. A common split is to use 70% of the data for training and 30% of the data for testing. We used 70% of our time series as a training data set and then split the remaining 30% as a validation set and testing set. The validation set is used to tune parameters to optimize every model and then the testing set is used to correctly evaluate the model performance. 

### Data Augmentation
Generate new data points by averaging the next day's forecasted value with the previous day's true data value.

This is a simple but effective data augmentation technique that can be used to increase the size of the training data set without having to collect new data. It is valid because it creates new data points that are representative of the real-world data. By averaging the next day's forecasted value with the previous day's true data value, we are creating new data points that reflect the natural variation in freight demand.

### Model Training
Train the neural network models on the augmented training data set.

Neural networks are a type of machine learning model that can be trained to learn complex patterns in data. They are well-suited for freight demand forecasting because they can learn the relationships between the various factors that influence freight demand.

### Use parameter tuning optimization based on MAPE to find the best parameters for each model.

Parameter tuning is the process of adjusting the hyperparameters of a machine learning model to improve its performance. MAPE (mean absolute percentage error) is a common metric used to evaluate the performance of forecasting models.

### Evaluate the performance of the trained models on the testing set.

This is done to assess how well the models are able to generalize to new data. The model with the lowest MAPE on the testing set is selected for forecasting.

### Model Selection
Select the neural network model that performs best on the testing set.

This model is the most likely to be able to make accurate forecasts on new data.

### Forecasting
Use the selected neural network model to forecast freight demand for the next day or week.

The trained model can be used to forecast freight demand for any future period. The forecasts can be used to make decisions about resource allocation, routing, and pricing.

### Why is data augmentation valid for freight demand forecasting?

Data augmentation is valid for freight demand forecasting because it creates new data points that are representative of the real-world data. By averaging the next day's forecasted value with the previous day's true data value, we are creating new data points that reflect the natural variation in freight demand. This helps the neural network model to learn more complex patterns in the data and to make more accurate forecasts.

In addition, data augmentation can help to reduce the risk of overfitting. Overfitting occurs when a machine learning model learns the training data too well and is unable to generalize to new data. By increasing the size and diversity of the training data set with data augmentation, we can help to reduce the risk of overfitting.

### Results and Discussion

In this section, we present the results of our experiments and discuss the implications of our findings.

Despite the limited length of our data set, we found that data augmentation significantly improved the performance of the neural network models for freight demand forecasting.

This improvement in performance suggests that neural network models are now a better alternative for freight demand forecasting than conventional forecasting methods. Neural network models are able to learn more complex patterns in the data, which allows them to make more accurate forecasts, especially when the data set is small.

In the following sections, we will visually analyze the results of our experiments and discuss their implications in more detail.

### Discussion

The results of our study suggest that data augmentation is a promising technique for improving the performance of neural network models for freight demand forecasting. Even with a relatively small data set, we were able to achieve significant improvements in the accuracy of the forecasts.

Our findings have important implications for the trucking industry. By using neural network models with data augmentation, trucking companies can develop more accurate freight demand forecasts. This can help them to improve their efficiency and reduce costs by enabling them to better allocate resources, optimize routes, and set prices.

Of course, there are some limitations to our study. First, our data set was relatively small, so we cannot guarantee that our findings will generalize to other data sets. Second, we only evaluated three types of neural network models. It is possible that other types of neural network models could perform even better with data augmentation.

Future research should focus on evaluating the performance of data augmentation with different types of neural network models and on larger data sets. Additionally, future research could explore the use of data augmentation for other forecasting tasks, such as retail demand forecasting and energy demand forecasting.
## Citations

```{bibliography}
```


