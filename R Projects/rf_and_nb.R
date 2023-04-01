##### Homework 3
##### Brandon Kimball


##### Question 6
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

set.seed(13)
cancer.df <- read_csv("/Users/brandonk87/Downloads/R Class/Naive Bayes/BreastCancer.csv")
cancer.df <- cancer.df %>% drop_na()
indx <- sample(nrow(cancer.df),nrow(cancer.df)*0.8)
train.df <- cancer.df[indx,]
test.df <- cancer.df[-indx,]

## 1. Naive Bayes Classifier Model
model.nb <- naive_bayes(as.factor(diagnosis)~., data=train.df)

predictions.nb <- predict(model.nb, test.df)

confusion.nb <- confusionMatrix(predictions.nb,
                                as.factor(test.df$diagnosis),
                                mode = "everything")
confusion.nb

## 2. Decision Tree Classifier Model
model.dt <- rpart(as.factor(diagnosis)~.,
                  data=train.df)

predictions.dt <- predict(model.dt, test.df, type = "class")
confusion.dt <- confusionMatrix(predictions.dt,
                as.factor(test.df$diagnosis),
                mode = "everything")
confusion.dt

model.dt$variable.importance/sum(model.dt$variable.importance)

## 3. Random Forest Classifier Model
model.rf <- randomForest(as.factor(diagnosis)~.,
                  data=train.df)

predictions.rf <- predict(model.rf, test.df, type = "class")
confusion.rf <- confusionMatrix(predictions.rf,
                as.factor(test.df$diagnosis),
                mode = "everything")
confusion.rf


## The algorithm with the best accuracy is the random forest algorithm (0.9115),
## The decision tree accuracy was only 0.9027 and the naive bayes accuracy was
## only 0.876
confusion.dt
confusion.rf
confusion.nb

## The algorithm with the best recall is the random forest algorithm (0.8298),
## The decision tree recall was only 0.809 and the naive bayes recall was
## only 0.766
confusion.dt
confusion.rf
confusion.nb

## Based on the decision tree, the top three most important features are 
## mean area (0.319), mean radius (0.314), and mean perimeter (0.3003).
model.dt$variable.importance/sum(model.dt$variable.importance)


##### Question 7
## Having developed three algorithms that can predict on a new diagnosis and 
## seeing which features are most important in determining the diagnosis status,
## I would advise all patients on the things to look out for and potential 
## methods of prevention based on the important features highlighted by the 
## decision tree model. This could also be used to determine which features are 
## necessary to screen patients on in hopes of catching the cancer early, instead
## of using every test (such as mean_smoothness which had a vary low relative 
## importance score) in order to save the patients time and money.


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

