library(tidyverse)
library(caret)
library(Metrics)
library(naivebayes)
library(randomForest)
library(rpart)
library(DescTools)

# Importing the data
websites.df <- read_csv("/Users/brandonk87/Downloads/R Class/Midterm/websites_labelled.csv")
websites.df$label <- as.factor(websites.df$label)

# Filling in missing data
# I used mode imputation
colSums(is.na(websites.df))
table(websites.df$server_loc)
websites.df$server_loc[is.na(websites.df$server_loc)] <- 
  Mode(websites.df$server_loc, na.rm = TRUE)[1]
sum(is.na(websites.df))

# Basic Data exploration
colnames(websites.df)
dim(websites.df)
str(websites.df)
head(websites.df)
View(websites.df)
table(websites.df$label)
table(websites.df$registered_domain)
table(websites.df$https)
table(websites.df$server_loc)
table(websites.df$most_visitors_loc)

# Practical Plots
ggplot(websites.df, aes(label, fill=label)) + geom_bar() + 
  labs(title="Number of Good vs. Bad Websites")
ggplot(websites.df, aes(x="", fill=label)) + geom_bar() +
  coord_polar(theta = "y") + labs(title = "Label distribution")
ggplot(websites.df, aes(registered_domain, fill=label)) + geom_bar() + 
  labs(title="Domain Registration Completion")
ggplot(websites.df, aes(https, fill=label)) + geom_bar() + 
  labs(title="https Usage")
ggplot(websites.df, aes(server_loc, fill=label)) + geom_bar() + 
  labs(title="Location of Server")
ggplot(websites.df, aes(most_visitors_loc, fill=label)) + geom_bar() + 
  labs(title="Location of Users")
ggplot(websites.df, aes(server_loc, fill=most_visitors_loc)) + geom_bar() + 
  labs(title="Server Location vs User Location")


# Splitting into train and test set so we can build the models
set.seed(13)
indx <- sample(nrow(websites.df), nrow(websites.df)*0.8)
train.df <- websites.df[indx,]
test.df <- websites.df[-indx,]

# Creating a dataframe to hold the metrics of the optimal models
AllMetrics <- data.frame()


# Change 1- Random forest on original data
rf.original <- randomForest(label~., data=train.df)
pred_prob <- predict(rf.original, test.df, type= "prob")
method <- list("Original")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test.df$label))
  confusion <- confusionMatrix(pred_factor, test.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
original.rf.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(original.rf.metrics) <- seq(1, 10)
original.rf.metrics
AllMetrics <- rbind(AllMetrics, original.rf.metrics[9, ])




# Change 2- Creating new columns into a df names train.newcol.df
## first will be a dummy column for server_loc where 1=america and 0=not america
nrow(filter(train.df, (server_loc == "Americas")))/nrow(train.df)
train.newcol.df <- train.df %>%
  mutate(Americas_server_loc = case_when(server_loc == "Americas" ~ 1,
                                         TRUE ~ 0))
## second will be a 1 if domain is both registered AND it used HTTPS
nrow(filter(train.df, (registered_domain == "complete"),(https == "yes")))/nrow(train.df)
train.newcol.df <- train.newcol.df %>%
  mutate(both_domain_and_https = case_when(registered_domain == "complete" &
                                             https == "yes" ~ 1,TRUE ~ 0))
## third will be a 1 if neither domain is registered OR used HTTPS
nrow(filter(train.df, (registered_domain == "incomplete"),(https == "no")))/nrow(train.df)
train.newcol.df <- train.newcol.df %>%
  mutate(neither_domain_or_https = case_when(registered_domain == "incomplete" &
                                               https == "no" ~ 1,TRUE ~ 0))
## fourth will be dummy column where 1= .com and 0= not a .com website
nrow(filter(train.df, (website_domain == "com")))/nrow(train.df)
train.newcol.df <- train.newcol.df %>%
  mutate(.com_website = case_when(website_domain == "com" ~ 1,
                                  TRUE ~ 0))
## fifth will be column binning unique users per day based on quartile
summary(train.df$unique_users_day)
train.newcol.df <- train.newcol.df %>%
  mutate(users_per_day_bin = case_when(unique_users_day <= 107 ~ "Bin1",
                                       unique_users_day <= 299 ~ "Bin2",
                                       unique_users_day <= 654 ~ "Bin3",
                                       unique_users_day > 654 ~ "Bin4"))

## Applying same column additions to the test dataframe
nrow(filter(test.df, (server_loc == "Americas")))/nrow(test.df)
test.newcol.df <- test.df %>%
  mutate(Americas_server_loc = case_when(server_loc == "Americas" ~ 1,TRUE ~ 0))

nrow(filter(test.df, (registered_domain == "complete"),(https == "yes")))/nrow(test.df)
test.newcol.df <- test.newcol.df %>%
  mutate(both_domain_and_https = case_when(registered_domain == "complete" &
                                             https == "yes" ~ 1,TRUE ~ 0))

nrow(filter(test.df, (registered_domain == "incomplete"),(https == "no")))/nrow(test.df)
test.newcol.df <- test.newcol.df %>%
  mutate(neither_domain_or_https = case_when(registered_domain == "incomplete" &
                                               https == "no" ~ 1,TRUE ~ 0))

nrow(filter(test.df, (website_domain == "com")))/nrow(test.df)
test.newcol.df <- test.newcol.df %>%
  mutate(.com_website = case_when(website_domain == "com" ~ 1, TRUE ~ 0))

summary(test.df$unique_users_day)
test.newcol.df <- test.newcol.df %>%
  mutate(users_per_day_bin = case_when(unique_users_day <= 107 ~ "Bin1",
                                       unique_users_day <= 299 ~ "Bin2",
                                       unique_users_day <= 654 ~ "Bin3",
                                       unique_users_day > 654 ~ "Bin4"))


# Change 2- Random forest on data with new columns added
rf.newcol <- randomForest(label~., data=train.newcol.df)
pred_prob <- predict(rf.newcol, test.newcol.df, type= "prob")
method <- list("New_Col")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test.newcol.df$label))
  confusion <- confusionMatrix(pred_factor, test.newcol.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
newcol.rf.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(newcol.rf.metrics) <- seq(1, 10)
newcol.rf.metrics
AllMetrics <- rbind(AllMetrics, newcol.rf.metrics[8, ])



# Change 3- Trying feature selection

# First method is feature selection off the original data and rf model
importance.rf.original <- tail(as.data.frame(rf.original$importance) %>% arrange(MeanDecreaseGini),5)
rownames(importance.rf.original)
rfimportance.original.df <- train.df[,c('label','registered_domain','unique_id','https','js_obf_len','js_len')]
rf.original.imp <- randomForest(label~., data=rfimportance.original.df)
pred_prob <- predict(rf.original.imp, test.df, type= "prob")
method <- list("Importance_Original")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test.df$label))
  confusion <- confusionMatrix(pred_factor, test.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
original.importance.rf.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(original.importance.rf.metrics) <- seq(1, 10)
original.importance.rf.metrics
AllMetrics <- rbind(AllMetrics, original.importance.rf.metrics[10, ])


# Second method is feature selection off the newcol data and rf model
importance.rf.newcol <- tail(as.data.frame(rf.newcol$importance) %>% arrange(MeanDecreaseGini),5)
rownames(importance.rf.newcol)
rfimportance.newcol.df <- train.newcol.df[,c('label','registered_domain','neither_domain_or_https','https','js_obf_len','js_len')]
rf.newcol.imp <- randomForest(label~., data=rfimportance.newcol.df)
pred_prob <- predict(rf.newcol.imp, test.newcol.df, type= "prob")
method <- list("Importance_Newcol")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test.newcol.df$label))
  confusion <- confusionMatrix(pred_factor, test.newcol.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
newcol.importance.rf.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(newcol.importance.rf.metrics) <- seq(1, 10)
newcol.importance.rf.metrics
AllMetrics <- rbind(AllMetrics, newcol.importance.rf.metrics[10, ])


# Third method is backward recursive elimination off the original data
control <- rfeControl(functions = rfFuncs, method = "repeatedcv",repeats = 5, verbose = FALSE)
rf.rfe.original <- randomForest(label~., data=train.df, rfeControl='control')
importance.rfe.original <- varImp(rf.rfe.original, scale=FALSE)
importance.rfe.original <- tail(importance.rfe.original %>% arrange(Overall),5)
rownames(importance.rfe.original)
rfrfe.original.df <- train.df[,c('label','registered_domain','https','unique_id','js_obf_len','js_len')]
rf.original.rfe <- randomForest(label~., data=rfrfe.original.df)
pred_prob <- predict(rf.original.rfe, test.df, type= "prob")
method <- list("Rfe_Original")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test.df$label))
  confusion <- confusionMatrix(pred_factor, test.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
original.rfe.rf.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(original.rfe.rf.metrics) <- seq(1, 10)
original.rfe.rf.metrics
AllMetrics <- rbind(AllMetrics, original.rfe.rf.metrics[10, ])


set.seed(13)
# Fourth method is backward recursive elimination off the newcol data
control <- rfeControl(functions = rfFuncs, method = "repeatedcv", repeats = 5, verbose = FALSE)
rf.rfe.newcol <- randomForest(label~., data=train.newcol.df, rfeControl='control')
importance.rfe.newcol <- varImp(rf.rfe.newcol, scale=FALSE)
importance.rfe.newcol <- tail(importance.rfe.newcol %>% arrange(Overall),5)
rownames(importance.rfe.newcol)
rfrfe.newcol.df <- train.newcol.df[,c('label','registered_domain','https',
                                      'neither_domain_or_https','js_obf_len','js_len')]
rf.newcol.rfe <- randomForest(label~., data=rfrfe.newcol.df)
pred_prob <- predict(rf.newcol.rfe, test.newcol.df, type= "prob")
method <- list("Rfe_NewCol")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test.newcol.df$label))
  confusion <- confusionMatrix(pred_factor, test.newcol.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
newcol.rfe.rf.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(newcol.rfe.rf.metrics) <- seq(1, 10)
newcol.rfe.rf.metrics
AllMetrics <- rbind(AllMetrics, newcol.rfe.rf.metrics[10, ])





set.seed(13)
# Change 7- Naive Bayes on original data
train.nb.df <- subset(train.df, select = -c(ip_add, website_domain))
test.nb.df <- subset(test.df, select = -c(ip_add, website_domain))
nb.original <- naive_bayes(label~., data=train.nb.df)
pred_prob <- predict(nb.original, test.nb.df, type= "prob")
method <- list("NB.Original")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test.nb.df$label))
  confusion <- confusionMatrix(pred_factor, test.nb.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
original.nb.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(original.nb.metrics) <- seq(1, 10)
original.nb.metrics
AllMetrics <- rbind(AllMetrics, original.nb.metrics[10, ])


set.seed(13)
# Change 8- Naive Bayes on data with new columns added
train.nb.newcol.df <- subset(train.newcol.df, select = -c(ip_add, website_domain))
test.nb.newcol.df <- subset(test.newcol.df, select = -c(ip_add, website_domain))
nb.newcol <- naive_bayes(label~., data=train.nb.newcol.df)
pred_prob <- predict(nb.newcol, test.nb.newcol.df, type= "prob")
method <- list("NB.New_Col")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test.nb.newcol.df$label))
  confusion <- confusionMatrix(pred_factor, test.nb.newcol.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
newcol.nb.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(newcol.nb.metrics) <- seq(1, 10)
newcol.nb.metrics
AllMetrics <- rbind(AllMetrics, newcol.nb.metrics[9, ])






# We now have 8 potential models
# We will choose one of them to continue on with into the over/under sampling step
# The AllMetrics df has the optimal metrics for each of the 6 models
AllMetrics
AllMetrics[which.max(AllMetrics$precision), ]
ggplot(AllMetrics, aes(x = method, y = accuracy)) + geom_point() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+ 
  labs(title="Accuracy of each method")
ggplot(AllMetrics, aes(x = method, y = precision)) + geom_point() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+ 
  labs(title="Precision of each method")
# Based on the results, we will continue on with the model that was built on the
# df that contained the new columns (and not involved in feature selection)
# This data set will be renamed into train2.df and test2.df to make it easier to track
train2.df <- train.newcol.df
test2.df <- test.newcol.df
# Also, we will make a new DF called AllMetrics2 to track the metrics of the new
# models, and the baseline model
AllMetrics2 <- data.frame()
AllMetrics2 <- rbind(AllMetrics2, newcol.rf.metrics[8, ])



# Change 4- Dealing with class imbalance: Over and under sampling
table(train2.df$label)/nrow(train2.df)
table(train2.df$label)

# Trying to up sample
train.up <- upSample(x = train2.df %>% select(-label), y = train2.df$label)  %>%
  rename("label" = "Class")
rf.up <- randomForest(label~., data=train.up)
pred_prob <- predict(rf.up, test2.df, type= "prob")
method <- list("Up_Sample")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.5, 0.95, by = 0.05)
for (thresholds in seq(0.5, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test2.df$label))
  confusion <- confusionMatrix(pred_factor, test2.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
rf.up.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(rf.up.metrics) <- seq(1, 10)
rf.up.metrics
AllMetrics2 <- rbind(AllMetrics2, rf.up.metrics[2, ])

# Trying to down sample
train.down <- downSample(x = train2.df %>% select(-label), y = train2.df$label)  %>%
  rename("label" = "Class")
rf.down <- randomForest(label~., data=train.down)
pred_prob <- predict(rf.down, test2.df, type= "prob")
method <- list("Down_Sample")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(0.50, 0.95, by = 0.05)
for (thresholds in seq(0.50, 0.95, by = 0.05)) {
  pred_classes <- ifelse(pred_prob[, "good"]> thresholds, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test2.df$label))
  confusion <- confusionMatrix(pred_factor, test2.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
rf.down.metrics <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(rf.down.metrics) <- seq(1, 10)
rf.down.metrics
AllMetrics2 <- rbind(AllMetrics2, rf.down.metrics[1, ])




# It is now time to pick an optimal random forest model out of the 3 left
AllMetrics2
ggplot(AllMetrics2, aes(x = method, y = accuracy)) + geom_point() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+ 
  labs(title="Accuracy of each method")
ggplot(AllMetrics2, aes(x = method, y = precision)) + geom_point() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+ 
  labs(title="Precision of each method")
# Based on the results, we will continue on with the model that was built on the
# df that contained the new columns. Up and down sampling did not improve the metrics



# Now we will hyper tune the parameters to try and optimize the model. 
# First is the mtry parameter. 
set.seed(13)
filter.tune.rf <- randomForest(label~., data=train2.df)
tune_results <- tuneRF(train2.df %>% select(-label), train2.df$label, stepFactor = 1.5)
tune_results

# Second is the ntree parameter. 
method <- list("NTree Tune")
precision <- list()
accuracy <- list()
falsepositive <- list()
threshold <- seq(100, 1000, by = 25)
for (ntrees in seq(100, 1000, by = 25)) {
  filter.tune.rf <- randomForest(label~., data=train2.df, mtry = 4, ntree = ntrees)
  pred_prob <- predict(filter.tune.rf, test2.df, type= "prob")
  threshold <- 0.85 # Use the optimal threshold which was determined earlier
  pred_classes <- ifelse(pred_prob[, "good"]> threshold, "good", "bad")
  pred_factor <- factor(pred_classes, levels = levels(test2.df$label))
  confusion <- confusionMatrix(pred_factor, test2.df$label, positive = "good", mode = "everything")
  precision <- append(precision, confusion$byClass["Precision"])
  accuracy <- append(accuracy, confusion$overall["Accuracy"])
  falsepositive <- append(falsepositive, confusion$table["good", "bad"])
}
ntree.tune <- cbind(method, accuracy, precision, falsepositive, threshold)
rownames(ntree.tune) <- seq(100, 1000, by = 25)
ntree.tune


# Model using optimal parameters (mtry = 4 and ntree = 475)
filter.tune.rf <- randomForest(label~., data=train2.df, mtry = 4, ntree = 475)
pred.tune.rf <- predict(filter.tune.rf, test2.df, type= "prob")
threshold <- 0.85 # Use the optimal threshold which was determined earlier
pred_classes <- ifelse(pred.tune.rf[, "good"]> threshold, "good", "bad")
pred_factor <- factor(pred_classes, levels = levels(test2.df$label))
confusion.tune.rf <- confusionMatrix(pred_factor, test2.df$label, positive = "good", mode = "everything")
confusion.tune.rf$table
confusion.tune.rf$table["good", "bad"]
confusion.tune.rf$overall["Accuracy"]
confusion.tune.rf$byClass["Precision"]


# Comparing the metrics of the non tuned model to the tuned model
newcol.rf.metrics[8,]
confusion.tune.rf$table["good", "bad"]
confusion.tune.rf$overall["Accuracy"]
confusion.tune.rf$byClass["Precision"]



# Have to apply the new columns to the original websites df in order to use this
# in the final random forest model
website.newcol.df <- websites.df %>%
  mutate(Americas_server_loc = case_when(server_loc == "Americas" ~ 1,
                                         TRUE ~ 0))

website.newcol.df <- website.newcol.df %>%
  mutate(both_domain_and_https = case_when(registered_domain == "complete" &
                                             https == "yes" ~ 1,TRUE ~ 0))

website.newcol.df <- website.newcol.df %>%
  mutate(neither_domain_or_https = case_when(registered_domain == "incomplete" &
                                               https == "no" ~ 1,TRUE ~ 0))

website.newcol.df <- website.newcol.df %>%
  mutate(.com_website = case_when(website_domain == "com" ~ 1,TRUE ~ 0))

website.newcol.df <- website.newcol.df %>%
  mutate(users_per_day_bin = case_when(unique_users_day <= 107 ~ "Bin1",
                                       unique_users_day <= 299 ~ "Bin2",
                                       unique_users_day <= 654 ~ "Bin3",
                                       unique_users_day > 654 ~ "Bin4"))


# Hypertuning the parameters leads to the best metrics yet, so we will use this 
# as our final model
# Testing final model 50 times in a cross validation of 5 folds
set.seed(13)
control <- trainControl(method = "repeatedcv", number = 5, repeats = 50)
final.model <- randomForest(label~., data=website.newcol.df, 
                            mtry = 4, ntree = 475, trControl = control)

TP <- final.model$confusion["good", "good"]
TN <- final.model$confusion["bad", "bad"]
FP <- final.model$confusion["bad", "good"]
FN <- final.model$confusion["good", "bad"]
# Accuracy
(TP + TN)/(TP+TN+FP+FN)
# Precision
TN/(TN+FN)




# Making prediction on the unlabeled data set
websites.unlabeled.df <- read_csv("/Users/brandonk87/Downloads/R Class/Midterm/websites_unlabelled.csv")
# We have to make the new columns in this data set before we can make predictions
# since our final model was based on the new columns
websites.unlabeled.df <- websites.unlabeled.df %>%
  mutate(Americas_server_loc = case_when(server_loc == "Americas" ~ 1,TRUE ~ 0))

websites.unlabeled.df <- websites.unlabeled.df %>%
  mutate(both_domain_and_https = case_when(registered_domain == "complete" &
                                             https == "yes" ~ 1,TRUE ~ 0))

websites.unlabeled.df <- websites.unlabeled.df %>%
  mutate(neither_domain_or_https = case_when(registered_domain == "incomplete" &
                                               https == "no" ~ 1,TRUE ~ 0))

websites.unlabeled.df <- websites.unlabeled.df %>%
  mutate(.com_website = case_when(website_domain == "com" ~ 1,TRUE ~ 0))

websites.unlabeled.df <- websites.unlabeled.df %>%
  mutate(users_per_day_bin = case_when(unique_users_day <= 107 ~ "Bin1",
                                       unique_users_day <= 299 ~ "Bin2",
                                       unique_users_day <= 654 ~ "Bin3",
                                       unique_users_day > 654 ~ "Bin4"))

# Now we make predictions
final_predictions <- predict(final.model, websites.unlabeled.df, type= "prob")
threshold <- 0.85 # Use the optimal threshold which was determined earlier
final_predictions <- ifelse(final_predictions[, "good"]> threshold, "good", "bad")
final_predictions <- factor(final_predictions, levels = levels(test2.df$label))
table(final_predictions)

