####### Homework 2
####### Brandon Kimball

### Question 1
library(tidyverse)
stroke <- read_csv("/Users/brandonk87/Downloads/R Class/Homework/strokedata.csv")
stroke <- na.omit(stroke)
sum(is.na(stroke))


### Question 2
#a
stroke_plot <- ggplot(data = stroke, aes(x=stroke)) + geom_histogram(bins=3)
stroke_plot
#b
age_plot <- ggplot(data = stroke, aes(x=age)) + geom_histogram()
age_plot


### Question 3
set.seed(123)
index <- sample(nrow(stroke),nrow(stroke)*0.85)
train <- stroke[index,]
test <- stroke[-index,]


### Question 4
#a
lrmodel1 <- glm(stroke~., data=train, family=binomial)
#b
summary(lrmodel1)
#c 
# The significant features are age, hypertension, and average glucose level


### Question 5
#a
predictions1 <- predict(lrmodel1, newdata = test, type = "response")
predictions1 <- ifelse(predictions1 >= 0.5, 1, 0)
predictions1
#b
table1 <- table(test$stroke,predictions1)
table1
sum(diag(table1))/sum(table1)
# The testing accuracy is rounded to 0.9498
# The reason it predicted 0 for all the test values is likely due to the 
# fact that the cutoff value is 0.5 instead of a lower value like 0.3.