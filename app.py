import streamlit as st
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
st.title('Credit Card Cluster Prediction, Characteristics and Schemes to be offered')

@st.cache_data
def display_df():
    info_data = pd.DataFrame()
    info_data['Column'] = ['CUST_ID','BALANCE','BALANCE_FREQUENCY','PURCHASES',
    'ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY',
    'CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS',
    'PRC_FULL_PAYMENT','TENURE']
    info_data['Column_info'] = ['Identification of Credit Card holder ',
    'Balance amount left in their account to make purchases', 
    'How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)',
    'Amount of purchases made from account',
    'Maximum purchase amount spent in one transaction',
    'Amount of purchase done in installment',
    'Cash in advance given by the user',
    'How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)',
    'How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)',
    'How frequently purchases in installments are being done (1 = frequently done,0 = not frequently done)',
    'How frequently the cash in advance being paid',
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

clustered_data = pd.read_csv('Clustered_data.csv', index_col=0)
cluster0 = clustered_data[clustered_data['KMC_clusters']==0].head(1)
cluster1 = clustered_data[clustered_data['KMC_clusters']==1].head(1)
cluster2 = clustered_data[clustered_data['KMC_clusters']==2].head(1)
cluster3 = clustered_data[clustered_data['KMC_clusters']==3].head(1)
st.info('Sample data for checking if the model works accurately')
st.info('Can input anything inplace of CUST_ID')
clustered_data = pd.concat([cluster0,cluster1,cluster2,cluster3])
clustered_data
cols = list(info_df['Column'].unique())

CUST_ID = st.text_input(
    label='Customer Id', 
    key='CUST_ID', 
    placeholder='e.g. C0000'
    )

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

PURCHASES_FREQUENCY = st.slider(
    label='Purchase Frequency',
    key='PURCHASES_FREQUENCY', 
    min_value=0.00, 
    max_value=1.00
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

CASH_ADVANCE_FREQUENCY = st.slider(
    label='Cash Advance Frequency', 
    key='CASH_ADVANCE_FREQUENCY',
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
import joblib

def load_models():
    model_preprocessor = joblib.load('model_preprocessor.pkl')
    model = joblib.load('pipeline_model.pkl')
    return model_preprocessor, model

model_preprocessor, model = load_models()
records = [CUST_ID, BALANCE,BALANCE_FREQUENCY,PURCHASES, ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,
CASH_ADVANCE,PURCHASES_FREQUENCY,ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
CASH_ADVANCE_FREQUENCY,	CASH_ADVANCE_TRX,	PURCHASES_TRX,	CREDIT_LIMIT,	PAYMENTS,
MINIMUM_PAYMENTS,PRC_FULL_PAYMENT,	TENURE]

show_df = st.checkbox('Show Input Data Table')
if show_df:
    input_data = pd.DataFrame([records], columns=cols)
    st.write(input_data)

with open('cluster_info.txt', 'r')as f:
    info = f.read()
clusterwise_info = info.split('@')

if st.button('PREDICT'):
    processed_data = model_preprocessor.transform(input_data)
    prediction = model.predict(processed_data)
    st.header('The given customer belongs to "Cluster  '+str(prediction[0])+'"')

    current_cluster = clusterwise_info[prediction[0]+1]
    st.header(current_cluster.split('$')[0])
    st.header(current_cluster.split('$')[1])


# st.cache
# st.cache_data
# st.cache_resource
