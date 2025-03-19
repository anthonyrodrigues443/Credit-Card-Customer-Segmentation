import streamlit as st
import pandas as pd
import pickle
import warnings
from preprocessor import preprocessor

warnings.filterwarnings('ignore')
st.title('Credit Card Cluster Prediction, Characteristics and Schemes to be offered')

@st.cache_data
def display_df():
    info_data = pd.DataFrame()
    info_data['Column'] = ['BALANCE','BALANCE_FREQUENCY','PURCHASES',
    'ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE',
    'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS',
    'PRC_FULL_PAYMENT','TENURE']
    info_data['Column_info'] = [
    'Balance amount left in their account to make purchases', 
    'How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)',
    'Amount of purchases made from account',
    'Maximum purchase amount spent in one transaction',
    'Amount of purchase done in installment',
    'Cash in advance given by the user',
    'How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)',
    'How frequently purchases in installments are being done (1 = frequently done,0 = not frequently done)',
    'Number of Transactions made with "Cash in Advanced"',
    'Number of purchase transactions made',
    'Limit of Credit Card for user',
    'Amount of Payment done by user',
    'Minimum amount of payments made by user',
    'Percent of full payment paid by user',
    'Tenure of credit card service for user']
    return info_data

info_df = display_df()
st.dataframe(info_df)

clustered_data = pd.read_csv('datasets/clustered_data.csv', index_col=0)
cluster0 = clustered_data[clustered_data['KMC_clusters']==0].head(1)
cluster1 = clustered_data[clustered_data['KMC_clusters']==1].head(1)
cluster2 = clustered_data[clustered_data['KMC_clusters']==2].head(1)
cluster3 = clustered_data[clustered_data['KMC_clusters']==3].head(1)
cluster4 = clustered_data[clustered_data['KMC_clusters']==4].head(1)
st.info('Sample data for checking if the model works accurately')

clustered_data = pd.concat([cluster0,cluster1,cluster2,cluster3, cluster4  ])
clustered_data
cols = list(info_df['Column'].unique())

BALANCE = st.number_input(
    label='Balance',
    key='BALANCE' , 
    min_value=0.00,
    placeholder='Enter customer balance'
    )

BALANCE_FREQUENCY = st.slider(
    label='Balance frequency',
    key='BALANCE_FREQUENCY',
    min_value=0.00,
    max_value=1.00
    )

PURCHASES = st.number_input(
    label='Purchases',
    key='PURCHASES',
    min_value=0.00,
    placeholder='Enter amount of purchases made from account'
    )

ONEOFF_PURCHASES = st.number_input(
    label='One Off Purchases', 
    key='ONEOFF_PURCHASES', 
    min_value=0.00, 
    placeholder='Maximum purchase amount spent in one transaction.'
    )

INSTALLMENTS_PURCHASES = st.number_input(
    label='Installments Purchases',
    key='INSTALLMENTS_PURCHASES',
    min_value=0.00,
    placeholder='Enter amount of installment purchases made from account'
    )

CASH_ADVANCE = st.number_input(
    label='Cash Advance', 
    key='CASH_ADVANCE',
    min_value=0.00,
    placeholder='Cash in advance given by the user'
    )

ONEOFF_PURCHASES_FREQUENCY = st.slider(
    label='One-Off Purchase Frequency',
    key='ONEOFF_PURCHASES_FREQUENCY', 
    min_value=0.00, 
    max_value=1.00
    )

PURCHASES_INSTALLMENTS_FREQUENCY = st.slider(
    label='Purchase Installments Frequency',
    key='PURCHASES_INSTALLMENTS_FREQUENCY',
    min_value=0.00,
    max_value=1.00
    )

CASH_ADVANCE_TRX = st.number_input(
    label='Cash Advance Transactions',
    key='CASH_ADVANCE_TRX',
    min_value=0,
    placeholder='Number of cash advance transactions')

PURCHASES_TRX = st.number_input(
    label='Purchase Transactions', 
    key='PURCHASES_TRX',
    min_value=0,
    placeholder='Number of purchase transactions')

CREDIT_LIMIT = st.number_input(
    label='Credit Limit', 
    key='CREDIT_LIMIT',
    min_value=0.0,
    placeholder='Enter customer credit limit'
    )

PAYMENTS = st.number_input(
    label='Payments', 
    key='PAYMENTS',
    min_value=0.0,
    placeholder='Enter total payments made'
    )

MINIMUM_PAYMENTS = st.number_input(
    label='Minimum Payments', 
    key='MINIMUM_PAYMENTS',
    min_value=0.0,
    placeholder='Enter minimum payments made'
    )

PRC_FULL_PAYMENT = st.slider(
    label='Percentage of Full Payment', 
    key='PRC_FULL_PAYMENT',
    min_value=0.00,
    max_value=1.00
    )

TENURE = st.slider(
    label='Tenure ',
    min_value=0,
    max_value=12
    )


def load_models():
    with open('models/PCA_model.pkl', 'rb')as pca_file:
        pca_model = pickle.load(pca_file)
    with open('models/KMeans_model.pkl', 'rb')as kmeans_model_:
        clustering_model = pickle.load(kmeans_model_)
    return pca_model, clustering_model

pca_model, clustering_model = load_models()
records = [ BALANCE,BALANCE_FREQUENCY,PURCHASES, ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,CASH_ADVANCE,
           ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,CASH_ADVANCE_TRX,	
           PURCHASES_TRX,	CREDIT_LIMIT,	PAYMENTS, MINIMUM_PAYMENTS,PRC_FULL_PAYMENT, TENURE]

show_df = st.checkbox('Show Input Data Table')
if show_df:
    input_data = pd.DataFrame([records], columns=cols)
    st.write(input_data)

with open('cluster_info.txt', 'r')as f:
    info = f.read()
clusterwise_info = info.split('@')

if st.button('PREDICT'):
    processed_df = preprocessor(input_data)
    pca_transformed_df = pca_model.transform(processed_df)
    prediction = clustering_model.predict(pca_transformed_df)
    st.header('The given customer belongs to "Cluster  '+str(prediction[0])+'"')
    current_cluster = clusterwise_info[prediction[0]+1]
    st.header(current_cluster)