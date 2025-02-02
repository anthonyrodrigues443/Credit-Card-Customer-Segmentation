<pre>
<h1>
Project : Credit Card Customer Segmentation
Author : Anthony Rodrigues
</h1>
</pre>

<pre>
1)Importing libraries - I've Imported all the libraries including libraries for preprocessing, visualization, 
statistical libraries, clustering models, cluster evaluation metrics, classification models, evaluation metrics, pipeline libraries.

2)Data ingestion, cleaning and preprocessing - Dropping the identifier column, dropping duplicate records, handling missing data 
using Simple Imputer, Normalization using log, dropping constant columns, handling multicollinearity, PCA for reducing dimensionality

3)EDA - Histograms to  visualize the distribution of null data columns to decide which statistical value to fill null with, to 
visualize feature distribution, pairplot along with lower and upper bounds to view outliers in each plot, heatmap to view correlation 
among features, line plots for Cumulative Explained Variance, Explained variance ratio to get the optimal number of principal
components, line plots for wcss and silhoutte score for obtaining the clusters number, Countplots to view cluster distribution, 
Scatterplots to view clusters of data points on PC's, 3dscatterplot to get better visualization of the, Dendogram to get the optimal 
number of clustering for Agglomerative clustering, pie charts to summarize percentages of each cluster, histograms to view where 
does the density of clusters lie, boxplots to view the statistical distribution of each cluster, scatterplots, kdeplots, 
clusterwise median plots to understand the clusters better.

4)Cluster model building and prediction - I've used two clustering techniques to cluster the data and to visualize the
clusters for each of the algorithms, decided to keep the KMeans Clusters as the clusters had higher silhoutte score and 
the clusters were better to understand and had better differenciating factors .

5)Cluster Analysis - After clustering I performed some clusterwise visualizations to analyze and understand the clusters
effectively and fetched important characterstics and personalized schemes for the customers.

7)Cross validation - Used Stratified Kfold to get the best model among all splits.

8)Classification Model building, training and evaluating - Intializing Boosting models (Decision tree, Random Forest, XGB, Gradient 
Boosting, XGBRF), Logistic Regression and SVC. Training with no parameter tuning . Evaluating models on unseen test data. Using 
Accuracy score as evaluation metrics(As got the best results no need for hyperparameter tuning).

9)Building Preprocessing and Model Pipelines - coded all the steps used in preprocessing steps (Clustering and Classification) in a 
python file imported the functions put in a pipeline, Similar approach for the models initialized the Kmeans and Decision tree model
in Pipeline ,trained the pipeline tested on data and saved in pickle file format.

10)Website Building - For Website I used streamlit which is easy to use, easy to deploy, and still has better UI. The website takes 
input for each feature of the data frame and puts displays the information for confirmation and predicts the cluster and the clusters 
characteristics and suitable schemes for the customer which was observedd through cluster analysis.


# Credit-Card-Customer-Segmentation
A machine learning solution for segmenting credit card customers based on spending patterns, payment behavior, and demographics to enable targeted marketing strategies and personalized services.
