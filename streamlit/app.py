import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import lime
import dill
import numpy as np
import plotly.express as px
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,recall_score,precision_score,auc
from mail_sender import sender


st.set_page_config(
    page_title="ChurnShield",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded")

with st.sidebar:
    st.image('../CHURNSHIELD.gif')
    # st.video('../CHURNSHIELD.mp4')
    st.title('🏂 ChurnShield')
    st.sidebar.header('Navigation')
    selected_option = st.sidebar.radio('',('Upload Dataset','Churn Analysis','Customer Behaviour Analysis', 'Prediction', 'Suggestion'))

# file = 0

    
def upload_dataset():
    st.title('')
    file = st.file_uploader('Upload dataset:')
    st.subheader('☁ Please upload data for the churn prediction')
    columns = ['account length','international plan','voice mail plan','number vmail messages',
                'total day minutes','total day calls','total day charge','total eve minutes',
                'total eve calls','total eve charge','total night minutes','total night calls',
                'total night charge','total intl minutes','total intl calls','total intl charge','customer service calls']
    
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv('dataset.csv',index=None)
        # print(df.columns.to_list())
        if all(elem in columns for elem in df.columns.to_list()):
            st.dataframe(df)
        # else:
        # st.warning(f'Need data to contain the followinf features : {columns}')
        xgboost_model = joblib.load('../models/telecom_syriatel/xgboost.pkl')
        X_test = df.iloc[:,:-1]
        y_test = df.iloc[:,-1]
        predictions = xgboost_model.predict(X_test)
        acc = accuracy_score(y_test,predictions)
        recall = recall_score(y_test,predictions)
        prec = precision_score(y_test,predictions)
        # auc_sc = auc(y_test,predictions)
        st.write('The models prediction metrics are:',acc )
        print(predictions)
        
        


def perform_current_analysis():
    df_tcst = pd.read_csv('../telecom_churn_data_syriatel.csv')
    churn_rate_by_state = df_tcst.groupby('state')['churn'].mean().reset_index()
    churn_rate_by_state['Churn Rate'] = churn_rate_by_state['churn'] * 100  # Convert to percentage


    st.header('📈 Visualising Churn rate')
    # Step 4: Create a Chart
    fig_state = px.bar(churn_rate_by_state, x='state', y='Churn Rate',
                title="Churn Rate by State",
                labels={"Churn Rate": "Churn Rate (%)", "State": "State"},
                color='Churn Rate',
                color_continuous_scale=px.colors.sequential.Viridis,
                )
    st.plotly_chart(fig_state,use_container_width=True)

    fig_map = px.choropleth(churn_rate_by_state,
                    locations='state',  # Use state abbreviations or full names
                    locationmode="USA-states",
                    color='Churn Rate',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    scope="usa",
                    title="Churn Rate by State in the USA",
                    width=800,
                    height=700
                    )
    st.plotly_chart(fig_map,use_container_width=True)

    churn_rate_by_calls = df_tcst.groupby('customer service calls')['churn'].mean().reset_index()
    churn_rate_by_calls['Churn Rate'] = churn_rate_by_calls['churn'] * 100  # Convert to percentage
    fig_cc = px.bar(churn_rate_by_calls, x='customer service calls', y='Churn Rate',
                title="Churn Rate by Number of Customer Service Calls",
                labels={"Churn Rate": "Churn Rate (%)", "Customer_service_calls": "Customer Service Calls"},
                text='Churn Rate',height=700)
    fig_cc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_cc,use_container_width=True)












# Function to make predictions based on user input
def predict_churn():
    st.header('🤖 Client Churn Predictor')
    st.subheader('Select any row from the follwing uploaded data to predict the churn for: ')
    # if file != 0:
    df = pd.read_csv('dataset.csv')
    st.dataframe(df)

    # Let the user select an index
    row_index = st.number_input('Select row index for prediction', min_value=0, max_value=len(df)-1, value=0, step=1)

    # Get the selected row
    selected_row = df.iloc[row_index]
    st.write('Selected Row for Prediction:', selected_row)
    
    x = selected_row[:-1].to_numpy().reshape(1, -1)
    y = selected_row.iloc[-1]
    
    print(x)
    print(y)
        
    sub = st.button('Submit')
    if sub:
        model = joblib.load('../models/telecom_syriatel/xgboost.pkl')
        out = model.predict(x)
        st.session_state['curr_inp'] = x
        if out == 0:
            text = 'The selected customer is predicted to not churn 🤩'
            st.write(out)
            st.markdown(f"<h1 style='text-align: center; font-size: 30px;'>{text}</h1>", unsafe_allow_html=True)
        else:
            text = 'The selected customer is predicted to churn 😢'
            st.write(out)
            st.markdown(f"<h1 style='text-align: center; font-size: 30px;'>{text}</h1>", unsafe_allow_html=True)
    # else:
    #     st.warning('Upload dataset first')
    # Threshold for unique values (as a percentage of the total rows)
    # unique_value_threshold = 0.5  # 50%

    # Calculate actual threshold value
    # threshold_value = unique_value_threshold * len(df)

    # Automatically select columns with datatype 'object' or 'category'
    # categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # # Filter out columns with unique values exceeding the threshold
    # useful_categorical_cols = [col for col in categorical_cols if df[col].nunique() <= threshold_value]

    # cols_to_remove = [col for col in categorical_cols if df[col].nunique() > unique_value_threshold]

    # # Remove these columns from the DataFrame
    # df = df.drop(columns=cols_to_remove)

    # label_encoder = LabelEncoder()

    # # Identify binary columns: columns with exactly 2 unique values
    # binary_cols = [col for col in df.columns if df[col].nunique() == 2]

    # # Apply Label Encoding to binary columns
    # for col in binary_cols:
    #     df[col] = label_encoder.fit_transform(df[col])

    
    # uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    # if uploaded_file:
    #     input_df = pd.read_csv(uploaded_file)
    #     st.write(
    #     '''
    #     ### Input Data ({} Customers)
    #     '''.format(input_df.shape[0])
    #     )
    #     st.dataframe(input_df)
    #     st.write('')
    #     rfm = pickle.load( open( "../Project 3/ran_forest_mod.p", "rb" ) )

    #     X = input_df.drop(labels = ['CustomerId'], axis = 1)

    #     threshold = .22
    #     y_preds = rfm.predict(X)
    #     predicted_proba = rfm.predict_proba(X)
    #     y_preds = (predicted_proba [:,1] >= threshold).astype('int')
    #     op_list = []
    #     for idx, exited in enumerate(y_preds):
    #         if exited == 1:
    #             op_list.append(input_df.CustomerId.iloc[idx])
    #     st.write('''### Number of Potentially Churning Customers''')
    #     st.write('''There are **{} customers** at risk of closing their accounts.'''.format(len(op_list)))

    #     csv = pd.DataFrame(op_list).to_csv(index=False, header = False)
    #     b64 = base64.b64encode(csv.encode()).decode()
    #     st.write('''''')
    #     st.write('''''')
    #     st.write('''### **⬇️ Download At-Risk Customer Id's**''')
    #     href = f'<a href="data:file/csv;base64,{b64}" download="at_risk_customerids.csv">Download csv file</a>'
    #     st.write(href, unsafe_allow_html=True)



def analysis():
    df_st = pd.read_csv('../customer_behaviour_analysis_refined.csv')
    st.header('📊 The Customer Analysis is performed on:')
    st.table(df_st.iloc[:50])

    # Distribution of Age
    fig_age = px.histogram(df_st, x='Age', title='Distribution of Age')
    # Distribution of Monthly Charge
    fig_monthly_charge = px.histogram(df_st, x='Monthly Charge', title='Distribution of Monthly Charge')
    # Churn by Gender
    fig_gender_churn = px.histogram(df_st, x='Gender', color='Customer Status', barmode='group', title='Churn by Gender')
    # Churn by Internet Service
    fig_internet_service_churn = px.histogram(df_st, x='Internet Service', color='Customer Status', barmode='group', title='Churn by Internet Service')

    # Layout with columns
    col1, col2 = st.columns(2)

    # Display charts in each column
    with col1:
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.plotly_chart(fig_monthly_charge, use_container_width=True)

    col3,col4 = st.columns(2)

    with col3:
        st.plotly_chart(fig_gender_churn, use_container_width=True)

    with col4:
        st.plotly_chart(fig_internet_service_churn, use_container_width=True)

    churned_customers_with_reason = df_st[df_st['Churn Reason'] != 'Not Specified']

    # Adjusting the visualization code to exclude the unsupported parameter
    fig_churn_reasons_adjusted = px.histogram(churned_customers_with_reason, x='Churn Reason', color='Churn Reason',
                                             height= 550,title='Churn reason')
    fig_churn_reasons_adjusted.update_layout(xaxis={'categoryorder':'total descending'}, showlegend=False)
    fig_churn_reasons_adjusted.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_churn_reasons_adjusted,use_container_width=True)


    # Prepare the data for mapping: select latitude, longitude, and churn status
    map_data = df_st[['Latitude', 'Longitude', 'Customer Status']]

    # Plotly map visualization
    st.subheader('Churn Distribution of customers')
    fig_map = px.scatter_mapbox(map_data, lat='Latitude', lon='Longitude', color='Customer Status',
                                color_discrete_map={"Churned": "red", "Stayed": "green"},
                                title='Customer Churn by Geographic Location',
                                zoom=5, height=600)

    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map,use_container_width=True)
    

def predict_fn(x):
    model = joblib.load('../models/telecom_syriatel/xgboost.pkl')
    out = model.predict_proba(x)
    return out


# Function to provide suggestions
def provide_suggestions():
    # Add code to provide suggestions based on analysis
    st.header('💡Suggestions Section')
    
    model_path = '../models/telecom_syriatel/lime_explainer.dill'

    # with open(model_path, 'rb') as file:
    #     lime_explainer = dill.load(file)
    lime_explainer = joblib.load(model_path)
        
    if 'curr_inp' not in st.session_state:
        st.warning('No customer selected')
        return
    x = st.session_state['curr_inp']
    explaination = lime_explainer.explain_instance(x, predict_fn, )
    
    explaination.as_pyplot_figure()
    name = st.text_input("Enter the recipient name: ")
    pos = st.text_input("Enter the position of recipient: ")
    contact = st.text_input("Enter the contact number: ")
    submit = st.button('Submit')
    if submit:
        sender(name,pos,contact)


def limeExplainer():
    df = pd.read_csv('dataset.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)






if selected_option == 'Customer Behaviour Analysis':
    analysis()
elif selected_option == 'Churn Analysis':
    perform_current_analysis()
elif selected_option == 'Upload Dataset':
    upload_dataset()
elif selected_option == 'Prediction':
    predict_churn()
    # Add widgets for user input (e.g., text input, sliders, dropdowns, file upload)
    # Display predictions based on user input
    # Example: st.write(predict_churn(user_input_data))
elif selected_option == 'Suggestion':
    provide_suggestions()
    
    

if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown(
        f'<style> .reportview-container .main .block-container{{max-width: 95%}} </style>',
        unsafe_allow_html=True,
    )