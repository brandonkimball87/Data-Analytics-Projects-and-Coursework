##### Homework 4
##### Brandon Kimball

library(tidyverse)
library(caret)
library(Metrics)
library(naivebayes)
library(randomForest)
library(rpart)
library(DescTools)

### Question 1
donations.df <- read_csv("/Users/brandonk87/Downloads/R Class/Input Engineering/donations.csv")
donations.df$EventAttendance2023 <- as.factor(donations.df$EventAttendance2023)
colSums(is.na(donations.df))
# I am going to fill the numeric columns with the column means and the
# character columns with the column mode
donations.df$Donations2020[is.na(donations.df$Donations2020)] <-
  mean(donations.df$Donations2020, na.rm = TRUE)
donations.df$Donations2021[is.na(donations.df$Donations2021)] <-
  mean(donations.df$Donations2021, na.rm = TRUE)
donations.df$Donations2022[is.na(donations.df$Donations2022)] <-
  mean(donations.df$Donations2022, na.rm = TRUE)
donations.df$Major[is.na(donations.df$Major)] <-
  Mode(donations.df$Major, na.rm = TRUE)[1]
sum(is.na(donations.df))


### Question 2
# Splitting data into 80/20 train and test sets
set.seed(13)
indx <- sample(nrow(donations.df), nrow(donations.df) * 0.8)
train.df <- donations.df[indx,]
test.df <- donations.df[-indx,]

# Creating  new columns
View(donations.df)
summary(donations.df$Donations2022)
# I am creating 5 new columns for each year if the donation amount is over $1,000
train.df.modif <- train.df %>%
  mutate(Donations2022_over1000 = case_when(Donations2022 >1000 ~ "yes",
                                            Donations2022 <= 1000 ~ "no"),
         Donations2021_over1000 = case_when(Donations2021 >1000 ~ "yes",
                                            Donations2021 <= 1000 ~ "no"),
         Donations2020_over1000 = case_when(Donations2020 >1000 ~ "yes",
                                            Donations2020 <= 1000 ~ "no"),
         Donations2019_over1000 = case_when(Donations2019 >1000 ~ "yes",
                                            Donations2019 <= 1000 ~ "no"),
         Donations2018_over1000 = case_when(Donations2018 >1000 ~ "yes",
                                            Donations2018 <= 1000 ~ "no"))
test.df.modif <- test.df %>%
  mutate(Donations2022_over1000 = case_when(Donations2022 >1000 ~ "yes",
                                            Donations2022 <= 1000 ~ "no"),
         Donations2021_over1000 = case_when(Donations2021 >1000 ~ "yes",
                                            Donations2021 <= 1000 ~ "no"),
         Donations2020_over1000 = case_when(Donations2020 >1000 ~ "yes",
                                            Donations2020 <= 1000 ~ "no"),
         Donations2019_over1000 = case_when(Donations2019 >1000 ~ "yes",
                                            Donations2019 <= 1000 ~ "no"),
         Donations2018_over1000 = case_when(Donations2018 >1000 ~ "yes",
                                            Donations2018 <= 1000 ~ "no"))

# Naive Bayes on original data
model.orig <- naive_bayes(EventAttendance2023~., 
                          data = train.df)
pred.orig <- predict(model.orig, test.df)

# Naive Bayes on modified data
model.modif <- naive_bayes(EventAttendance2023~., 
                          data = train.df.modif)
pred.modif <- predict(model.modif, test.df.modif)

# Model comparison
confusionMatrix(pred.orig, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.modif, test.df$EventAttendance2023, mode = "everything")
# The prediction accuracy for the unmodified data is 0.1823. The pred accuracy
# for the modified data (modified predictions vs unmodified test predictions)
# is 0.1768. This is slightly lower than the original data and we can conclude
# that adding new features did not improve the prediction accuracy


### Question 3
# Two different feature selection methods
# First is backward recursive feature elimination
control <- rfeControl(functions = nbFuncs,
                      method = "repeatedcv", repeats = 5,verbose = FALSE)
backward.model <- train(EventAttendance2023~., data = train.df.modif, 
                        rfeControl="control")
backward.importance <- varImp(backward.model, scale = FALSE)
plot(backward.importance)
backward.importance
# Second is selection by filtering 
control <- sbfControl(functions = nbSBF, method = "repeatedcv", repeats = 5)
filter.model <- sbf(x = train.df.modif %>% select(-EventAttendance2023),
                    y = train.df.modif$EventAttendance2023, sbfControl = control)
filter.model$optVariables

#Creating new data frames which contain their respective top 5 features (The sbf
# filter method only had 2 features survive, so only those 2 are in that dataframe)
backward.train <- train.df.modif %>%
  select(Donations2022, Donations2021, Donations2020, Donations2019, Donations2018, EventAttendance2023)
filter.train <- train.df.modif %>%
  select(Donations2022, Donations2020, EventAttendance2023)

# building nb models for each of the new data frames
backward.nb.model <- naive_bayes(EventAttendance2023~., data = backward.train)
pred.backward.nb <- predict(backward.nb.model, test.df.modif)

filter.nb.model <- naive_bayes(EventAttendance2023~., data = filter.train)
pred.filter.nb <- predict(filter.nb.model, test.df.modif)

# Comparing the two feature selection models and original data
confusionMatrix(pred.backward.nb, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.filter.nb, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.orig, test.df$EventAttendance2023, mode = "everything")
# The backward recursive feature selection selected the top 5 features which were 
# all the donation years. Running a naive bayes on these 5 features and then predicting
# on the test data, the resulting accuracy was 0.1657. 
# The selection by filtering method only had 2 features survive (2022 and 2020 
# donations), and running naive bayes on these two features led to an accuracy 
# of 0.2155
# The original naive bayes model had an accuracy of 0.1823, so the selection by
# filtering method improved the accuracy, and the backward recursive feature 
# selection reduced the accuracy. This means we will continue on with the model
# and training dataset which came from the selection by filtering method.


### Question 4
table(filter.train$EventAttendance2023)
# creating an oversampled and undersampled dataframe
filter.over <- upSample(x = filter.train %>% select(-EventAttendance2023),
                        y = filter.train$EventAttendance2023)
filter.over <- filter.over %>% rename("EventAttendance2023" = "Class")
table(filter.over$EventAttendance2023)
filter.under <- downSample(x = filter.train %>% select(-EventAttendance2023),
                        y = filter.train$EventAttendance2023)
filter.under <- filter.under %>% rename("EventAttendance2023" = "Class")
table(filter.under$EventAttendance2023)

# naive bayes models on the over and undersampled datasets
filter.over.model <- naive_bayes(EventAttendance2023~., data = filter.over)
pred.filter.over <- predict(filter.over.model, test.df.modif)

filter.under.model <- naive_bayes(EventAttendance2023~., data = filter.under)
pred.filter.under <- predict(filter.under.model, test.df.modif)

# Comparing the two sampling models and original data
confusionMatrix(pred.filter.over, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.filter.under, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.orig, test.df$EventAttendance2023, mode = "everything")
# The accuracy for oversampling the data is 0.1492. The accuracy for undersampling
# the data is 0.1878. The accuracy for the original data is 0.1823. This means that 
# undersampling improved the accuracy compared to original data, while oversampling
# made the prediction accuracy worse.


### Question 5
# I am going to tune the laplace, usekernal and adjust
# This will be done using the modified under sampling data set
set.seed(13)
control <- trainControl(method = "cv", number = 5)

grid <- expand.grid(laplace = c(0,1,2,3,4),
                    usekernel = c(TRUE),
                    adjust = c(1, 1.25, 1.5, 1.75, 2))

# Model that has feature selection, undersampling, and hyperparameter tuning
filter.under.tune.model <- train(EventAttendance2023~., data = filter.under,
                                 method = "naive_bayes", 
                                 trControl = control,
                                 tuneGrid = grid,
                                 metric = "Accuracy")
filter.under.tune.model$results
filter.under.tune.model$bestTune
pred.filter.under.tune <- predict(filter.under.tune.model$finalModel, test.df.modif)
confusionMatrix(pred.filter.under.tune, test.df$EventAttendance2023, mode = "everything")

# Model that has feature selection and hyperparameter tuning
filter.tune.model <- train(EventAttendance2023~., data = filter.train,
                           method = "naive_bayes", 
                           trControl = control,
                           tuneGrid = grid,
                           metric = "Accuracy")
filter.tune.model$bestTune
pred.filter.tune <- predict(filter.tune.model$finalModel, test.df.modif)
confusionMatrix(pred.filter.tune, test.df$EventAttendance2023, mode = "everything")

# The accuracy of the model with feature selection, undersampling, and 
# hyperparameter tuning is 0.4309
# The accuracy of the model with just feature selection, and hyperparameter 
# tuning is 0.6464. The parameters selected here were laplace of 0, usekernal of
# TRUE, and adjust of 1.
# This means that the model with just feature selection and hyper parameter
# tuning has provides the best prediction accuracy on the response variable so far


### Question 6

# The best model so far is the one that has the donations 2020 and donations 2022
# features and hyper tuned the three possible naive_bayes parameters where
#laplace of 0, usekernal of TRUE, and adjust of 1 are most optimal
# We will use this as the final model and run a 5 fold cross validation 50 
# times to get the final accuracy prediction.

set.seed(13)
grid.final <- expand.grid(laplace = 0, usekernel = c(TRUE),adjust = 1)
control <- trainControl(method = "repeatedcv", number = 5, repeats = 50)
final.model <- train(EventAttendance2023~., data = filter.train,
                     method = "naive_bayes", trControl = control, 
                     tuneGrid = grid.final, metric = "Accuracy")
final.model$results
# After determining the optimal parameters and feature selection, and running cross
# validation 50 times, the results say that on average, the prediction accuracy in
# determining the correct EventAttendance2023 outcome is 0.6456.



########### Bonus question ############
# new model testing
# Naive Bayes on original data
rf.orig <- randomForest(EventAttendance2023~., 
                          data = train.df)
pred.orig <- predict(rf.orig, test.df)
# Naive Bayes on modified data
rf.modif <- randomForest(EventAttendance2023~., 
                           data = train.df.modif)
pred.modif <- predict(rf.modif, test.df.modif)
# Model comparison
confusionMatrix(pred.orig, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.modif, test.df$EventAttendance2023, mode = "everything")
## Using the modified df which contains the extra columns gives a better accuracy
# score of 0.922 compared to original model which is 0.9061

# Feature selection
# doing backwards recursive elimination
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv", repeats = 5,verbose = FALSE)
backward.rf <- train(EventAttendance2023~., data = train.df.modif, 
                        rfeControl="control")
backward.importance <- varImp(backward.rf, scale = FALSE)
plot(backward.importance)
backward.importance
# doing filtering
control <- sbfControl(functions = rfSBF, method = "repeatedcv", repeats = 5)
filter.rf <- sbf(x = train.df.modif %>% select(-EventAttendance2023),
                    y = train.df.modif$EventAttendance2023, sbfControl = control)
filter.rf$optVariables
# The backwards selection picked the same top 5 features as was selected when naive
# bayes was ran, so I will reuse the same df 
# The filter method also selected the 2 features as was selected for nb earlier,
# so I will also reuse that specific df

#backward model building
backward.rf <- randomForest(EventAttendance2023~., data = backward.train)
pred.backward.rf <- predict(backward.rf, test.df.modif)
#filter model building
filter.rf <- randomForest(EventAttendance2023~., data = filter.train)
pred.filter.rf <- predict(filter.rf, test.df.modif)

# Comparing the two feature selection models and original data
confusionMatrix(pred.backward.rf, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.filter.rf, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.orig, test.df$EventAttendance2023, mode = "everything")

# The backward feature selection predicted at a 0.917 accuracy rate
# The filter method predicted at a 0.922 accuracy rate
# This means I will continue on with using the features selected by filtering

# Over/Under sampling
table(filter.train$EventAttendance2023)
# creating an oversampled and undersampled dataframe
filter.over <- upSample(x = filter.train %>% select(-EventAttendance2023),
                        y = filter.train$EventAttendance2023)
filter.over <- filter.over %>% rename("EventAttendance2023" = "Class")
table(filter.over$EventAttendance2023)
filter.under <- downSample(x = filter.train %>% select(-EventAttendance2023),
                           y = filter.train$EventAttendance2023)
filter.under <- filter.under %>% rename("EventAttendance2023" = "Class")
table(filter.under$EventAttendance2023)

# rf models on the over and undersampled datasets
filter.over.rf <- randomForest(EventAttendance2023~., data = filter.over)
pred.filter.over.rf <- predict(filter.over.rf, test.df.modif)

filter.under.rf <- randomForest(EventAttendance2023~., data = filter.under)
pred.filter.under.rf <- predict(filter.under.rf, test.df.modif)

# Comparing the two sampling models and original data
confusionMatrix(pred.filter.over.rf, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.filter.under.rf, test.df$EventAttendance2023, mode = "everything")
confusionMatrix(pred.orig, test.df$EventAttendance2023, mode = "everything")
# The accuracy for oversampling the data is 0.5746. The accuracy for undersampling
# the data is 0.4254. The accuracy for the original data is 0.9061. This means that 
# both sampling methods made it worse, so we will just use unsampled data

# Hypertuning parameters: mtry, splitrule, min.node.size
set.seed(13)
control <- trainControl(method = "cv", number = 5)

grid <- expand.grid(mtry = c(1,2),
                    min.node.size = c(1,2,3,4,5,6,7,8,9),
                    splitrule = c("extratrees"))

filter.tune.rf <- train(EventAttendance2023~., data = filter.train,
                           method = "ranger", 
                           trControl = control,
                           tuneGrid = grid)
filter.tune.rf$bestTune
pred.filter.tune.rf <- predict(filter.tune.rf, test.df.modif)
confusionMatrix(pred.filter.tune.rf, test.df$EventAttendance2023, mode = "everything")
# Hypertuning the parameters leads to an accuracy of 0.9337. This is the best
# accuracy yet, so we will use this as out final model

# Testing final model 50 times in a cross validation of 5 folds
set.seed(13)
grid.final <- expand.grid(mtry = 1, min.node.size = 4, splitrule = c("extratrees"))
control <- trainControl(method = "repeatedcv", number = 5, repeats = 50)
final.model <- train(EventAttendance2023~., data = filter.train,
                     method = "ranger", trControl = control, 
                     tuneGrid = grid.final)
final.model$results
# After determining the optimal parameters and feature selection, and running cross
# validation 50 times, the results say that on average, the prediction accuracy in
# determining the correct EventAttendance2023 outcome is 0.8753. This is higher than
# the final model made by naive bayes, so I would recommend to my boss to use
# this model which was built using random forest