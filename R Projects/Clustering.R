library(tidyverse)
library(gridExtra)
library(datawizard)
library(factoextra)
library(dbscan)
library(cluster)


### Question 2
# Load the dataset
candybars.df <- read_csv("/Users/brandonk87/Downloads/R Class/Clustering/CandyBars.csv")
# Preprocess data for clustering
candybars.df <- candybars.df %>% select(-Brands)
colSums(is.na(candybars.df))
candy.scaled <- as.data.frame(normalize(candybars.df))
# K Means
kmeans1 <- kmeans(candy.scaled, centers = 4)
fviz_cluster(kmeans1, data = candy.scaled, geom="point", ellipse.type = "convex")
sum(kmeans1$withinss)
final.kmeans <- fviz_nbclust(candy.scaled, FUNcluster = kmeans, method = "wss")
final.kmeans 
# We will continue with an optimal k value of 5, as this is the point where the 
# wss begins to level out
kmeans.optimal <- kmeans(candy.scaled, centers = 5)
sum(kmeans.optimal$withinss)
candy.clusters.kmean <- candybars.df %>% mutate(Kmeans = kmeans.optimal$cluster)



### Question 3
candy.clusters.kmean %>% split(.$Kmeans) %>% map(summary)
candy.clusters.kmean %>%
  gather(key = Features, values, -Kmeans) %>%
  ggplot(aes(x=Features, y=values,)) + 
  geom_boxplot(show.legend=FALSE) + facet_wrap(~Kmeans) +
  labs(title="Candybar Kmeans Boxplots") +
  theme_bw() + ylim(0, 300) + coord_flip()
# Cluster 1 is called "High Calories". This is because the mean is 284 and the 
# min is 273 (which is higher than the maximum calories of any other group)
# Cluster 2 is called "High Protein". This is because it has a high
# calorie count but has an abnormally high protein count compared to all other 
# groups (mean is 5.5)
# Cluster 3 is called "High Carb". This is because the other groups carb means 
# are between 28-37 but this groups mean is 56.
# Cluster 4 is called "Low Calories". This is because the mean is 166 and the 
# max is 212 (which is lower than the maximum calories of most other group)
# Cluster 5 is called "Average". This is because the mean values for this group
# are very similar to the overall mean of all candy bars in this data frame


### Question 4
kNNdistplot(candy.scaled, k=3)
# We will continue with an epsilon of 0.23 which is the value before an 
# exponential increase in epsilon distance. Additionally, a minPts of 4 led to the
# highest silhouette score
dbscan1 <- dbscan(candy.scaled, eps = 0.23, minPts = 4)
table(dbscan1$cluster)
fviz_cluster(dbscan1, data = candy.scaled, geom="point", ellipse.type = "convex")
candy.clusters.dbscan <- candybars.df %>% mutate(dbscan = dbscan1$cluster)



### Question 5
candy.clusters.dbscan %>% split(.$dbscan) %>% map(summary)
candy.clusters.dbscan %>%
  gather(key = Features, values, -dbscan) %>%
  ggplot(aes(x=Features, y=values,)) + 
  geom_boxplot(show.legend=FALSE) + facet_wrap(~dbscan) +
  labs(title="Candybar DBSCAN Boxplots") +
  theme_bw() + ylim(0, 300) + coord_flip()
# Cluster 0 is called "Scattered Nutrition".
# Cluster 1 is called "Precise Nutrition". 
# The reason these names were chosen is because for cluster 1, all of the features
# are in a confined value range. For example, the range of calories is only 99, 
# compared to 172 in cluster 0. If you choose a candy bar at random from this 
# range, you have a good idea the nutrients it will contain, but if you did that
# for cluster 0, the nutrients could be all over the place.



### Question 6
sil.kmeans.score <- silhouette(kmeans.optimal$cluster, dist(candybars.df))
sil.kmeans.result <- summary(sil.kmeans.score)
sil.kmeans.result$avg.width

sil.dbscan.score <- silhouette(dbscan1$cluster, dist(candybars.df))
sil.dbscan.result <- summary(sil.dbscan.score)
sil.dbscan.result$avg.width

# The silhouette score for the kmeans assignments was 0.17. This value is very low
# and suggests that this method did not find a significant structure. The silhouette
# score for the dbscan assignments was 0.189. This value is also extremely low, 
# meaning the clusters are overlapping and not in their proper assignment. The dbscan
# score was slightly higher than the kmeans score, meaning that method did a better
# job with cluster assignment, but overall, still not significant
