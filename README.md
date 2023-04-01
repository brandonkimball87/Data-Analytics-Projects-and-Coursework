# Data Analytics Portfilio

## About

Hi, my name is Brandon Kimball. I am currently in the final semester of the Masters of Science in Analytics program at The University of Alabama Huntsville (May 2023 graduation).
I have two undergraduate degrees (BS in Biology and BA in Psychology) from The University of Alabama in Tuscaloosa.

The majority of my work is in R and Python, in addition to SQL. The major focus of my projects and coursework involves predictive analytics and machine learning. Specific algorithms include regression, clustering, CART, Random Forest, Naïve Bayes and various ensemble methods. Each of the examples below contain a file and description which showcases a homework, project or exam assignment where a certain machine learning or data analytics technique was used.

## Table of Contents
[**About**](#about)

[**Python**](#python)  
- [Logistic Regression- Coronary Heart Disease Prediction](#coronary-heart-disease-prediction)  
- [Linear Regression- Housing Prices Prediction](#housing-prices-prediction)
- [Time Series Analysis](#time-series-analysis)
- [Clustering: KNN and KMeans](#clustering-knn-and-kmeans)
- [Regularization: LASSO and Ridge](#regularization-lasso-and-ridge)
- [CART Based Analysis](#cart-based-analysis)
- [Ensemble Methods for Machine Learning](#ensemble-methods-for-machine-learning)

[**R**](#r)  
-  [Malicious Webpages Case Study](#malicious-webpages-case-study)
-  [Donations Prediction](#donations-prediction)
-  [Breast Cancer Prediction](#breast-cancer-prediction)
-  [Car Prices Prediction](#car-prices-prediction)
-  [Logistic Regression- Stroke Prediction](#stroke-prediction)



## Python

### Coronary Heart Disease Prediction
**Skills**:  
**Code**: [Coronary_Heart_Disease_Prediction.ipynb](./Python%20Projects/Coronary%20Heart%20Disease%20Prediction.ipynb)    
**Description**:  

### Housing Prices Prediction
**Skills**:  
**Code**: [Housing_Prices_Prediction.ipynb](./Python%20Projects/Housing%20Prices%20Prediction.ipynb)  
**Description**:    

### Time Series Analysis
**Skills**: ARMA model, AR/MA term hyper tuning, QQ and ACF/PCF plots     
**Code**: [Time_Series_Analysis.ipynb](./Python%20Projects/Time_Series_Analysis.ipynb)  
**Description**: The task for this project was to make a prediction for the number of daily female births in CA for the next 30 days. The first step was to construct ACF and PCF plots to determine the most appropriate ARMA model. This was followed by training the model and explaining the significance for each of the AR/MA terms included. Diagnostic checks were conducted by creating and exploring the sequence plot, histogram plot, Q-Q plot, and ACF plot of the residuals. After hyper tuning the model by adding an additional AR term or MA term, the final model was used for prediction. 

### Clustering: KNN and KMeans
**Skills**: KNN, KMeans, k hyperparameter tuning, elbow method heuristic, parametric vs non-parametric algorithms
**Code**: [KNN.ipynb](./Python%20Projects/KNN.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;[KMeans.ipynb](./Python%20Projects/KMeans.ipynb)  
**Description**: The two clustering methods explored here include KNN and KMeans. The KNN file includes cross validation and training-testing validation to determine the optimal value of K for k-nearest neighbor. The KMeans file explores coordinate visualization and the elbow method heuristic for determining the optimal value for K. Both methods are used to predict the type of wheat based on a set of geometric parameters of internal wheat kernel structure detected using a soft X-ray technique.

### Regularization: LASSO and Ridge
**Skills**:  LASSO, Ridge, lambda optimization  
**Code**: [Regularization.ipynb](./Python%20Projects/Regularization.ipynb)   
**Description**: This project uses regularization methods to predict the percentage of a state’s total counted vote that was for George Bush in the 2000 presidential election. The first method is LASSO variable selection (least absolute shrinkage and selection operator), which simultaneously estimates coefficients and preforms variable selection by adding a regularizer to the loss function. The second method is Ridge variable selection, which focuses  on multicollinearity, instead of feature selection. Both methods utilize a regularizer, called a lambda penalty, and my code shows how I preformed cross validation to hyper tune the lambda to its optimal value. This is how you find the balance between bias and variance, which prevents overfitting and creates an algorithm that can be applied to new, unseen data.    

### CART Based Analysis
**Skills**: DecisionTreeRegressor, DecisionTreeClassifier, cost-complexity pruning, alpha optimization  
**Code**: [CART.ipynb](./Python%20Projects/CART.ipynb)   
**Description**: Here I explore the use of a regression tree to predict total vote percentage in the 2000 presidential election. Also, a classification tree is used to predict bankruptcy using 10 predictors. Cost-complexity pruning was applied to both models in order to reduce variance. Specifically, cross validation was used to determine the optimal alpha, which is the penalty applied in order to prevent overfitting.   

### Ensemble Methods for Machine Learning
**Skills**: Bootstrap Aggregation, Adaboost, Gradient Boosting   
**Code**: [Bagging.ipynb](./Python%20Projects/Bagging.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;[Boosting.ipynb](./Python%20Projects/Boosting.ipynb)  
**Description**: The three attached files explore different ensemble methods in machine learning. The first examines the use of bagging (bootstrap aggregation) to optimize a random forest algorithm. Various techniques to finetune the model include out of bag (OOB) prediction and max feature optimization. The second file explores multiple Boosting techniques to create a model which can most accurately predict bankruptcy. The first technique is Adaboost (adaptive boosting), which adjusts the weights of different weak learners based on misclassification. The second technique is Gradient Boosting, which repeatedly fits a new weak learner based on the errors of the previous weak learners. Both methods were hyper tuned using Cross Validation to find the optimal parameters, such as number of trees and learning rate.


## R

### Malicious Webpages Case Study  
**Skills**:  
**Code**: [Malicious_Webpages_Case_Study.R](./R%20Projects/Malicious_Webpages_Case_Study.R)      
**Description**:  


### Donations Prediction
**Skills**:  
**Code**: [Donations_Prediction.R](./R%20Projects/Donations_Prediction.R)  
**Description**:  


### Breast Cancer Prediction
**Skills**:  
**Code**: [Breast_Cancer_Prediction.R](./R%20Projects/Breast_Cancer_Prediction.R)     
**Description**:  


### Car Prices Prediction
**Skills**:  
**Code**: [Car_Prices_Prediction.R](./R%20Projects/Car_Prices_Prediction.R)    
**Description**:  


### Stroke Prediction 
**Skills**:  
**Code**: [Stroke_Prediction.R](./R%20Projects/Stroke_Prediction.R)    
**Description**:  
