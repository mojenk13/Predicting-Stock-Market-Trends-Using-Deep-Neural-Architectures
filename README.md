# Predicting-Stock-Market-Trends-Using-Deep-Neural-Architectures

Predicting stock market trends is challenging due to the nonlinear nature of fi-
nancial time series data. In this project, we evaluate the forecasting performance of
a couple deep learning models: Convolutional Neural Networks (CNN), Variational
Autoencoders (VAE), and Radial Basis Function Extreme Learning Machines (RBF-
ELM), both individually and in combination with one another. Using a sliding window
approach on AAPL stock prices, we compare models based on Mean Absolute Error
(MAE) and Fluctuation Around the Trend (FAT). Among all, the standalone RBF-
ELM achieved the best results, with the lowest MAE of 1.53 and one of the lowest FAT
scores of 14.79. It also performed well on out-of-sample predictions with an MAE of
3.70. On the other hand, the combined models did not offer improvements and often
produced less stable forecasts, though they still captured the general form of predicted
trends. Our findings suggest that, in some cases, simpler models like RBF-ELM can
outperform more complex architectures for short-term stock prediction.
