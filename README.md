# Data Analytics Portfilio

## About

Hi, my name is Brandon Kimball. I am currently in the final semester of the Masters of Science in Analytics program at The University of Alabama Huntsville (May 2023 graduation).
I have two undergraduate degrees (BS in Biology and BA in Psychology) from The University of Alabama in Tuscaloosa.

The majority of my work is in R and Python, in addition to SQL. The major focus of my projects and coursework involves predictive analytics and machine learning. Specific algorithms include regression, clustering, CART, Random Forest, Naïve Bayes, SVM, various ensemble methods and text mining. Each of the examples below contain a file and description which showcases a homework, project or exam assignment where a certain machine learning or data analytics technique was used.

My semester long capstone project was to develop an algorithm which automates the creation of capability matrices based on incoming requirement documents for a leading company in the defense industry. This project was done using Python and includes machine learning and natural language methods such as Doc2Vec, NLTK, cosine similarity, and bag of words. Ultimately, this capstone project was an incredible learning experience where I could apply the knowledge and skills gained throughout my master's program to develop a practical solution for a real-world problem.   

## Table of Contents
[**About**](#about)

[**Python**](#python)  
- [Natural Language Processing](#natural-language-processing)
- [Ensemble Methods for Machine Learning](#ensemble-methods-for-machine-learning)
- [Clustering: KNN and KMeans](#clustering-knn-and-kmeans)
- [Regularization: LASSO and Ridge](#regularization-lasso-and-ridge)
- [Support Vector Machine](#support-vector-machine)
- [CART Based Analysis](#cart-based-analysis)
- [Imbalanced Classification](#imbalanced-classification)  
- [Linear Regression](#linear-regression)
- [Time Series Analysis](#time-series-analysis)

[**R**](#r)  
-  [Malicious Webpages Case Study](#malicious-webpages-case-study)
-  [Text Mining](#text-mining)
-  [Clustering: KMeans and DBSCAN](#clustering-kmeans-and-dbscan)
-  [Input Engineering](#input-engineering)
-  [Random Forest and Naive Bayes](#random-forest-and-naive-bayes)
-  [Logistic Regression](#logistic-regression)



## Python

### Natural Language Processing
**Skills**: Doc2Vec, NLTK, cosine similarity, bag of words   
**Code**: [natural_language_processing.ipynb](./Python%20Projects/natural_language_processing.ipynb)   
**Description**: This is a modified version of the code for the natural language processing capstone project. This version compares scientific articles (focusing on a specific aspect of the kidney) to 8 general kidney physiology statements. Each article is broken down into individual paragraphs and common stop words are removed to provide more precise results. For each of the 8 general statement, the resulting top 5 most similar paragraphs, cosine similarity scores, and document names are provided and outputted to the Google Drive as a CSV. This provides a good example of the domain flexibility that natural language processing provides; from creating capability matrices for large businesses to analyzing the insights of renal studies in the scientific field.      

### Ensemble Methods for Machine Learning
**Skills**: Bootstrap Aggregation, Adaboost, Gradient Boosting   
**Code**: [Bagging.ipynb](./Python%20Projects/Bagging.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;[Boosting.ipynb](./Python%20Projects/Boosting.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;[Stacking.ipynb](./Python%20Projects/Stacking.ipynb)  
**Description**: The three attached files explore different ensemble methods in machine learning. The first examines the use of bagging (bootstrap aggregation) to optimize a random forest algorithm. Various techniques to finetune the model include out of bag (OOB) prediction and max feature optimization. The second file explores multiple Boosting techniques to create a model which can most accurately predict bankruptcy. The first technique is Adaboost (adaptive boosting), which adjusts the weights of different weak learners based on misclassification. The second technique is Gradient Boosting, which repeatedly fits a new weak learner based on the errors of the previous weak learners. Both methods were hyper tuned using Cross Validation to find the optimal parameters, such as number of trees and learning rate. The third file uses stacking techniques to find the optimal combination of various prediction algorithms. Base learners used are logistic regression, KNN, decision trees, random forest, gradient boosting, and regularization (lasso and ridge). The meta learner used is logistic regression.  

### Clustering: KNN and KMeans
**Skills**: KNN, KMeans, k hyperparameter tuning, elbow method heuristic, parametric vs non-parametric algorithms  
**Code**: [KNN.ipynb](./Python%20Projects/KNN.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;[KMeans.ipynb](./Python%20Projects/KMeans.ipynb)  
**Description**: The two clustering methods explored here include KNN and KMeans. The KNN file includes cross validation and training-testing validation to determine the optimal value of K for k-nearest neighbor. The KMeans file explores coordinate visualization and the elbow method heuristic for determining the optimal value for K. Both methods are used to predict the type of wheat based on a set of geometric parameters of internal wheat kernel structure detected using a soft X-ray technique.

### Regularization: LASSO and Ridge
**Skills**:  LASSO, Ridge, lambda optimization  
**Code**: [Regularization.ipynb](./Python%20Projects/Regularization.ipynb)   
**Description**: This project uses regularization methods to predict the percentage of a state’s total counted vote that was for George Bush in the 2000 presidential election. The first method is LASSO variable selection (least absolute shrinkage and selection operator), which simultaneously estimates coefficients and preforms variable selection by adding a regularizer to the loss function. The second method is Ridge variable selection, which focuses  on multicollinearity, instead of feature selection. Both methods utilize a regularizer, called a lambda penalty, and my code shows how I preformed cross validation to hyper tune the lambda to its optimal value. This is how you find the balance between bias and variance, which prevents overfitting and creates an algorithm that can be applied to new, unseen data.    

### Support Vector Machine
**Skills**: support vector machine, kernel functions, C hyper parameter tuning  
**Code**: [SVM.ipynb](./Python%20Projects/SVM.ipynb)   
**Description**: Utilizing SVM, the goal of this project was to predict if a banknote was real or fake. Multiple models were built which explored the various kernel functions (sbf, linear, polynomial) and hyper tuning the C regularization parameter to find the balance between accuracy and margin maximization. Cross validation techniques led to a final SVM algorithm which had an overall prediction accuracy on the test set of 100% .         

### CART Based Analysis
**Skills**: DecisionTreeRegressor, DecisionTreeClassifier, cost-complexity pruning, alpha optimization  
**Code**: [CART.ipynb](./Python%20Projects/CART.ipynb)   
**Description**: Here I explore the use of a regression tree to predict total vote percentage in the 2000 presidential election. Also, a classification tree is used to predict bankruptcy using 10 predictors. Cost-complexity pruning was applied to both models in order to reduce variance. Specifically, cross validation was used to determine the optimal alpha, which is the penalty applied in order to prevent overfitting.   

### Imbalanced Classification
**Skills**: Oversampling, Undersampling, Cost Sensitive Learning, Multiple Stochastic Regression Based Imputation  
**Code**: [Imbalanced Classification.ipynb](./Python%20Projects/Imbalanced_Classification.ipynb)    
**Description**: The goal of this project was to predict if a patient will have coronary heart disease in the future. Because only 15% of the training set had a disease outcome, extra work has to be done to ensure this minority class has a proper representation when building the model. The first two methods were simply oversampling the minority class to the same level as the majority and undersampling the majority class to the same level as the minority class. The other two methods, cost sensitive learning and multiple stochastic regression based imputation, involve more complicated algorithms to handle the imbalanced classification. Metrics to examine model performance include precision, recall, F1, and AUC.  

### Linear Regression
**Skills**: Model diagnostics, correction, and assesment, Feature selection, Outliers and leverage, Missing data   
**Code**: [Linear_Regression.ipynb](./Python%20Projects/Linear_Regression.ipynb)  
**Description**: This was the final exam/project for a semester long course on python based data analytics. The goal was to predict housing prices based on certain factors such as house location, number of bedrooms, furnished, nearness to main road, etc. After adding dummy variables and handling missing data, matplotlib.pyplot was used to visualize the data. The independent variable then transformed using a third degree polynomial, log and Box Cox transformation. Residual plots were exaimined to detrmine the optimal transformation. Forward, backward and best subset selection techniques were used for feature selection. Finally, cross validation and training testing validation was preformed to estimate the generalization performance of the algorithm.   

### Time Series Analysis
**Skills**: ARMA model, AR/MA term hyper tuning, QQ and ACF/PCF plots     
**Code**: [Time_Series_Analysis.ipynb](./Python%20Projects/Time_Series_Analysis.ipynb)  
**Description**: The task for this project was to make a prediction for the number of daily female births in CA for the next 30 days. The first step was to construct ACF and PCF plots to determine the most appropriate ARMA model. This was followed by training the model and explaining the significance for each of the AR/MA terms included. Diagnostic checks were conducted by creating and exploring the sequence plot, histogram plot, Q-Q plot, and ACF plot of the residuals. After hyper tuning the model by adding an additional AR term or MA term, the final model was used for prediction. 


## R

### Malicious Webpages Case Study  
**Skills**: randomForest, feature selection, Naive Bayes, imbalanced classification, hyperparameter tuning, missing data, ggplot   
**Code**: [Malicious_Webpages_Case_Study.R](./R%20Projects/Malicious_Webpages_Case_Study.R)      
**Description**: The goal of this exam was to create an algorithm which can predict the status of a new website for a private security organization based on 36,623 training data observations. The overall process started with handling the missing data and observing trends using ggplot. Random Forest and naive bayes models were trained for each modification to the data set (adding new features, feature selection, and imbalanced classification on the independent variable). Finally, hyperparameter tuning helped create the final model recommendation. The assignment also factored in that from a business perspective, the worst outcome is predicting that a website is good, but in actuality, the website is bad. This required adjusting the algorithm to reduce the number of false positives and finding the proper balance between overall accuracy and precision.     


### Text Mining  
**Skills**: sentiment analysis, tokenization, stop words, topic modeling, term document matrix, latent dirichlet allocation, word cloud     
**Code**: [text_mining.R](./R%20Projects/text_mining.R)      
**Description**: This project analyzed the State of the Union addresses by various presidents throughout history. Pre-processing of data included tokenization and stop word removal. Sentiment analysis was then performed on each of George Washington’s State of the Union addresses using the AFINN and bing sentiment dictionary. This found the most positive and negative speech from the 8 he gave. The ncr sentiment dictionary was also used to find the single most positive and negative speech out of every president's State of the Union. A term document matrix and latent dirichlet allocation model was made based on these speeches. This led to the creation of 5 unique groups, each based on a specific topic. A word cloud was then constructed for the 500, 100, and 10 most common words in all of Abraham Lincoln’s State of the Union addresses.        


### Clustering: KMeans and DBSCAN
**Skills**: KMeans, DBSCAN, Silhouette score, Cluster labeling  
**Code**: [Clustering.R](./R%20Projects/Clustering.R)     
**Description**: The dataset used in this assignment was a list of candy bars and their nutrition values. The purpose was to apply unsupervised learning methods to group the candy bars and then use visualizations to name each cluster. The two algorithms were KMeans and DBSCAN (density based clustering), and silhouette scoring was used to quantify each model's performance.    


### Input Engineering
**Skills**: Feature Selection, Adding New Features, Class Imbalance, Missing Data, Hyperparameter Tuning    
**Code**: [Input_Engineering.R](./R%20Projects/Input_Engineering.R)  
**Description**:  The attached file shows the five feature engineering techniques used on predictive algorithms. The dataset here was about election donations and the goal was to predict if an individual would attend the campaign event in 2023. The first method is missing data imputation, including forward fill, backward fill and column mean/mode. Next is creating new columns and features. Some popular methods used here were one hot encoding and binning values, but also this was a chance to practice exploring the data, using creativity and applying domain knowledge to create new features which could improve prediction accuracy. After the new features were created, feature selection methods (backward recursive feature elimination and selection by filtering) picked out the significant columns that would be included in the final model. Next was dealing with imbalanced classification from the "event attendance" independent variable (upSample and downSample functions). Finally, the last step was to hyper tune parameters (laplace and usekernal for naive bayes and mtry, min.node.size, and splitrule for random forest) and run a 5 fold cross validation 50 times to create a final model recommendation.     


### Random Forest and Naive Bayes
**Skills**: Random Forest, Decision Tree and Naive Bayes classifiers/regresors  
**Code**: [rf_and_nb.R](./R%20Projects/rf_and_nb.R)    
**Description**: The first half of the assignment was to apply a naive Bayes classifier, decision tree classifier, and random forest classifier to create an algorithm for predicting breast cancer in patients. The second half of the assignment was to apply a naive Bayes regressor, decision tree regressor, and random forest regressor to create an algorithm for predicting car prices one year in the future.   


### Logistic Regression 
**Skills**: logistic regression, ggplot, train/test split, feature significance  
**Code**: [Logistic_Regression.R](./R%20Projects/Logistic_Regression.R)    
**Description**: This homework assignment was to work with dataset that provides details about people who had strokes and build a logistic regression model which can predict who is at risk for a stroke. Specific methods in this assignment involved visualization via ggplot, training/testing split, and determining the specific features (risk factors in this case) which are significant in determining a stroke outcome.   
