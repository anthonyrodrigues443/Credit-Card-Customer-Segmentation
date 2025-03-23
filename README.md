# Credit Card Customer Segmentation 💳👥📊 - Unsupervised ML
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-FF4B4B?logo=streamlit&logoColor=white)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-green.svg)](https://scikit-learn.org/)

## Overview 🔎
This project is a machine learning-based web application for segmenting credit card customers. It employs a robust data preprocessing pipeline and advanced clustering techniques to group customers based on their spending behavior and financial patterns. The application is built using Python, Streamlit for the web interface, and scikit-learn’s K-Means algorithm for customer segmentation.

## Aim 🎯
- Effectively segment credit card customers into distinct groups based on their spending behavior and financial patterns.
- Perform Exploratory Data Analysis (EDA) after clustering to uncover key characteristics of each customer segment.
- Identify potential personalized financial schemes tailored to the needs of each cluster.
- Provide valuable insights to financial institutions for customer retention, targeted marketing, and credit risk management.
- Develop a scalable, interpretable, and robust customer segmentation system.

### Webpage
<img src="https://github.com/anthonyrodrigues443/Credit-Card-Customer-Segmentation/blob/main/webpage_images/web1.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Credit-Card-Customer-Segmentation/blob/main/webpage_images/web2.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Credit-Card-Customer-Segmentation/blob/main/webpage_images/web3.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Credit-Card-Customer-Segmentation/blob/main/webpage_images/web4.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Credit-Card-Customer-Segmentation/blob/main/webpage_images/web5.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Credit-Card-Customer-Segmentation/blob/main/webpage_images/web6.png" width="400px">


## Project Structure 🗂️
```
Credit-Card-Customer-Segmentation/
├── customer_seg_proj/         # Python virtual environment
├── datasets/                  # Dataset directory
│   ├── cluster0_data.csv
│   ├── cluster1_data.csv
│   ├── cluster2_data.csv
│   ├── cluster3_data.csv
│   ├── cluster4_data.csv
│   ├── clustered_data.csv
│   ├── Customer Data.csv
├── models/                    # Trained clustering models
│   ├── KMeans_model.pkl        # K-Means clustering model
│   ├── PCA_model.pkl           # PCA model for dimensionality reduction
├── preprocessing_values/       # Preprocessing configurations and values
├── webpage_images/             # Images of the web page and terminal logging
├── __pycache__/                # Cached Python files
├── .gitignore                  # Git ignore file
├── app.py                      # Streamlit application entry point
├── cluster_info.txt            # Information about clusters characteristics and schemes
├── cust_seg_nb.ipynb           # Jupyter Notebook for model building
├── preprocessor.py             # Script for preprocessing data
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies

```

## Features ⭐


## Installation 🧑‍🔧
1. Clone the repository:
   ```bash
   git clone https://github.com/anthonyrodrigues443/Credit-Card-Customer-Segmentation.git
   cd Credit-Card-Customer-Segmentation
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 👨🏻‍💻
1. Start the Flask application:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to `http://localhost:8501`.
3. Enter applicant details in the provided form.
4. View the predicted cluster, characteristics of that cluster and schemes for the customer.

## Data Preprocessing Pipeline ⛓️


### Example Usage:
```python
#1. run the notebook file "cust_seg_nb.ipynb" to create the models

with open('models/PCA_model.pkl', 'rb')as pca_file:
    pca_model = pickle.load(pca_file)
with open('models/KMeans_model.pkl', 'rb')as kmeans_model_:
    clustering_model = pickle.load(kmeans_model_)

records = [ BALANCE,BALANCE_FREQUENCY,PURCHASES, ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,CASH_ADVANCE,
           ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,CASH_ADVANCE_TRX,	
           PURCHASES_TRX,	CREDIT_LIMIT,	PAYMENTS, MINIMUM_PAYMENTS,PRC_FULL_PAYMENT, TENURE]

input_data = pd.DataFrame([records], columns=cols)

processed_df = preprocessor(input_data)
pca_transformed_df = pca_model.transform(processed_df)
prediction = clustering_model.predict(pca_transformed_df)

print(prediction[0])
```

## Model Details 🤖


## Contributing 🤝
1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request.

## License 📋
This project is licensed under the MIT License. See the [LICENSE](https://github.com/anthonyrodrigues443/Credit-Card-Customer-Segmentation/blob/main/LICENSE) file for more details.

<h3>⭐ Don't forget to star the repository if you find it helpful!


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