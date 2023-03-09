
import numpy as np
import pickle
import pandas as pd
import streamlit as st
import datetime



def deploy():
  st.set_page_config(layout='wide')
  
  st.title('ITEM DEMAND FORECASTING')
  st.header('Enter the item Details')
  date=st.date_input('which date you want to Forecast :')
  item=st.number_input('which item you want to forecast:' , min_value=1,max_value=50)

  if st.button('Demand Forecast'):
    forecast=predict(date,item)
    forecast=np.round_(forecast,0)
    st.success(f'Demand of {item} Item at after 3 Months is : {forecast[0]}')
def isweekend(data):
  if data < 5 :
    return 0
  else:
    return 1
  
def predict(date,item):
  date=pd.to_datetime(date)
  duration=(pd.to_datetime(date+pd.DateOffset(months=+3))-pd.to_datetime(date)).days
  df=pd.DataFrame({'date':[date],'item':[item],'duration':[duration]})
  df['day']=df['date'].dt.day
  df['dayofyear']=df['date'].dt.dayofyear
  df['week']=df['date'].dt.weekofyear
  df['weekday']=df['date'].dt.weekday
  df['weekend']=[isweekend(i) for i in df['weekday']]
  df['month']=df['date'].dt.month
  df['quarter']=df['date'].dt.quarter
  df['year']=df['date'].dt.year
  df['days_count']=df['date'].dt.days_in_month.astype(int)
  df['month_start']=df['date'].dt.is_month_start.astype(int)
  df['month_end']=df['date'].dt.is_month_end.astype(int)
  df['year_start']=df['date'].dt.is_year_start.astype(int)
  df['year_end']=df['date'].dt.is_year_end.astype(int)
  df['quarter_start']=df['date'].dt.is_quarter_start.astype(int)
  df['quarter_end']=df['date'].dt.is_quarter_end.astype(int)
  X_test=df.drop(['date'],axis=1).values
  model=pickle.load(open('best_Xgb_model.pkl','rb'))
  prediction=model.predict(X_test)
    
  return prediction
  

if __name__=='__main__':
  deploy()
