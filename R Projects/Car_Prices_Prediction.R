library(tidyverse)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(naivebayes)
library(rpart)
library(Metrics)
library(randomForest)

##### Question 8
car.df <- read_csv("/Users/brandonk87/Downloads/R Class/Naive Bayes/CarPrice.csv")
car.df <- car.df %>% drop_na()

## 1. Decision Tree Regression Model
train_control <- trainControl(method="cv", number=3)

model.dt <- train(price_in_one_year~.,
                  data=car.df,
                  method="rpart",
                  trControl=train_control)
model.dt$results
predictions.dt <- predict(model.dt, car.df)

importance.dt <- varImp(model.dt)
importance.dt$importance/sum(importance.dt$importance)

mae(predictions.dt,car.df$price_in_one_year)
rmse(predictions.dt,car.df$price_in_one_year)

## 2. Random Forest Regression Model
train_control <- trainControl(method="cv", number=3)

model.rf <- train(price_in_one_year~.,
                  data=car.df,
                  method="rf",
                  trControl=train_control)
model.rf$results
predictions.rf <- predict(model.rf, car.df)

importance.rf <- varImp(model.rf)
importance.rf$importance/sum(importance.rf$importance)

mae(predictions.rf,car.df$price_in_one_year)
rmse(predictions.rf,car.df$price_in_one_year)

## The algorithm with the best MAE is the random forest model which is 606 compared
## to the decision tree MAE of 2244
mae(predictions.dt,car.df$price_in_one_year)
mae(predictions.rf,car.df$price_in_one_year)

## The algorithm with the best RMSE is is the random forest model which is 886
## compared to the decision tree RMSE of 2966
rmse(predictions.dt,car.df$price_in_one_year)
rmse(predictions.rf,car.df$price_in_one_year)

## Based on the decision tree, the top three most important features are 
## engine size (0.217), curb weight (0.209), and highway mpg (0.198)
importance.dt <- varImp(model.dt)
importance.dt$importance/sum(importance.dt$importance)


##### Question 9
## If you were a car dealership, you could look to stock the cars which have 
## more important features (like engine size) that lead to a higher year one price 
##in order to maximize the potential profit.
## If you were a car manufacturer, you could focus your resources on improving 
## those specific features which lead to a higher year one price, in hopes of
## maximizing profit.

