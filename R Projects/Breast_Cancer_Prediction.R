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



