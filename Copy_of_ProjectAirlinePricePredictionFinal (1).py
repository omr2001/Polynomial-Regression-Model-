#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


#Reading Data
Data=pd.read_csv("airline-price-prediction.csv")


# In[3]:


#Exploring Data
Data.head()


# In[4]:



Data.info()


# In[5]:


#Seeing Null Values and Duplicate Values
Data.isnull().sum()
Data=Data.drop_duplicates(subset=None,keep='first',inplace=False)


# In[6]:


#Exploring Shape of Data
Data.shape


# In[7]:


# a Copy of Data to Work on 
R_Data=Data.copy()


# In[8]:


R_Data.head(2)


# In[9]:


# Exploring Datatypes of Data
R_Data.dtypes


# In[10]:


#Turning Date from String to Datetime Function
def Change_Into_Datetime(col):
    R_Data[col]=pd.to_datetime(R_Data[col])


# In[11]:


R_Data.columns


# In[12]:


# Preprocessing Price by Turning it to Float 
R_Data['price'] = R_Data['price'].str.replace(',',"")
R_Data['price']
R_Data = R_Data.astype({'price':'float'})


# In[13]:


# Turning Date,Arrival and Departure Time into Datetime
for feature in ['date','dep_time','arr_time'] :
    Change_Into_Datetime(feature)


# In[14]:


R_Data.dtypes


# In[15]:


R_Data['date'].min()


# In[16]:


R_Data['date'].max()


# In[17]:


#Creating a Column of Journey Day
R_Data['Journey_Day']=R_Data['date'].dt.day


# In[18]:


##Creating a Column of Journey Month
R_Data['Journey_Month']=R_Data['date'].dt.month


# In[19]:


##Creating a Column of Journey Year
R_Data['Journey_Year']=R_Data['date'].dt.year


# In[20]:


R_Data.head(2)


# In[21]:


#Extrating Hours and Minutes Function
def extract_hour_min(Data,col):
    Data[col+"_hour"]=Data[col].dt.hour
    Data[col+"_Minute"]=Data[col].dt.minute
    Data.drop(col,axis=1,inplace=True)
    return Data.head(2)


# In[22]:


#Extrating Hours and Minutes 
extract_hour_min(R_Data,'dep_time')


# In[23]:


extract_hour_min(R_Data,'arr_time')


# In[24]:


#Exploring Departure Time
def Flight_Dep_Time(x):
    if(x>4) and (x<=8) : 
        return "Early Morning"
    elif (x>8) and (x<=12) : 
        return "Morning"
    elif (x>12) and (x<=16) : 
        return "Noon"
    elif (x>16) and (x<=20) : 
        return "Evening"
    elif (x>20) and (x<=24) :
        return "Night"
    else : 
        return "Late_Night"
        


# In[25]:


#Importing Important Libraries
import plotly 
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[26]:


cf.go_offline()


# In[27]:


# Visualizing Departure Time hours
R_Data['dep_time_hour'].apply(Flight_Dep_Time).value_counts().iplot(kind='bar')


# In[28]:


#Preprocess Duration Function
def Pre_Process_Duration(x):
    if 'h' not in x :
        x='0h '+x
    elif 'm' not in x:
        x=x+' 0m'
    return x


# In[29]:


#Applying on Time Taken
R_Data['time_taken']=R_Data['time_taken'].apply(Pre_Process_Duration)


# In[30]:


R_Data['time_taken']


# In[31]:


int(R_Data['time_taken'][0].split(' ')[1][0:-1])


# In[32]:


# Extracting The Hours taken
R_Data['time_taken_hours']=R_Data['time_taken'].apply(lambda x:int(float(x.split(' ')[0][0:-1])))
R_Data['time_taken_hours']


# In[33]:


#Extracting Minutes and setting '' values to 0
R_Data['time_taken_minutes']=R_Data.time_taken.str.split(" ",expand=True)[1]
R_Data['time_taken_minutes']=R_Data.time_taken_minutes.str.split("m",expand=True)[0]
R_Data['time_taken_minutes'].unique()
R_Data['time_taken_minutes'][91880]='32'
R_Data['time_taken_minutes'][127110]='20'
R_Data['time_taken_minutes'].unique()
R_Data['time_taken_minutes']=R_Data['time_taken_minutes'].astype(int)
R_Data['time_taken_minutes']
#R_Data['time_taken_minutes']=R_Data['time_taken_minutes'].str.replace('','0')
#R_Data['time_taken_minutes']=R_Data['time_taken_minutes'].astype(int)


# In[34]:


len(R_Data)


# In[35]:


#R_Data['time_taken_minutes']=R_Data['time_taken'].apply(lambda x:int(float(x.split('h')[1][0:-1])))
#R_Data['time_taken_minutes']


# In[36]:


#Assigning Total Time Taken in Minutes
R_Data['total_time_taken']=R_Data['time_taken_hours'].astype(float)*60+R_Data['time_taken_minutes'].astype(float)
R_Data['total_time_taken']


# In[37]:


R_Data.head(2)


# In[38]:


#Exploring Distribution of Total Time Taken based on Price
sns.lmplot(x='total_time_taken',y='price',data=R_Data)


# In[39]:


#Preprocessing Route
R_Data['route'].unique()


# In[40]:


#Splitting Source and Destination
R_Data[['source','destination']] = R_Data.route.str.split(",",expand=True)


# In[41]:


#Splitting Source Column
R_Data['source']=R_Data.source.str.split(":",expand=True)[1]
R_Data['source'] = R_Data['source'].astype(str)
R_Data['source']


# In[42]:


# Splitting Destination Column
R_Data['destination']=R_Data.destination.str.split(":",expand=True)[1]
R_Data['destination']=R_Data.destination.str.replace("}","")
R_Data['destination']


# In[43]:


#Plotting Violin Plot to See The Relation between Airline and Price
plt.figure(figsize=(15,5))
sns.violinplot(x='airline',y='price',data=R_Data)
plt.xticks(rotation='vertical')


# In[44]:


R_Data.drop(columns=['route','total_time_taken','Journey_Year'],axis=1,inplace=True)
#R_Data.drop(columns=['route','Journey_Year'],axis=1,inplace=True)


# In[45]:


#Applying Target encoding on Source
sources=R_Data.groupby(['source'])['price'].mean().sort_values().index
dict2={key:index for index,key in enumerate(sources,0)}
R_Data['source']=R_Data['source'].map(dict2)
R_Data['source']


# In[46]:


R_Data.head(3)


# In[47]:


#Applying Target guided encoding on Airline
airlines=R_Data.groupby(['airline'])['price'].mean().sort_values().index


# In[48]:


airlines


# In[49]:


dict1={key:index for index,key in enumerate(airlines,0)}


# In[50]:


R_Data['airline']=R_Data['airline'].map(dict1)


# In[51]:


R_Data['airline']


# In[52]:


R_Data.head(2)


# In[53]:


#Applying Target Guided Encoding on Destination
dest=R_Data.groupby(['destination'])['price'].mean().sort_values().index


# In[54]:


dest


# In[55]:


dict2={key:index for index,key in enumerate(dest,0)}


# In[56]:


dict2


# In[57]:


R_Data['destination']=R_Data['destination'].map(dict2)


# In[58]:


R_Data['destination']


# In[59]:


R_Data.head(2)


# In[60]:


#Extracting Meaningful features from Stop Column
R_Data['stop'].unique()


# In[61]:


total_stops={'non-stop':0, '2+-stop':2}


# In[62]:


R_Data['stop']=R_Data['stop'].map(total_stops)


# In[63]:


R_Data['stop']


# In[64]:


R_Data['stop']=R_Data['stop'].fillna(1)
R_Data['stop']


# In[65]:


#Detecting Outliers
def plot(R_Data,col):
    fig,(ax1,ax2,ax3)=plt.subplots(3,1)
    sns.distplot(R_Data[col],ax=ax1)
    sns.boxplot(R_Data[col],ax=ax2)
    sns.distplot(R_Data[col],ax=ax3,kde=False)


# In[66]:


plot(R_Data,'price')


# In[67]:


#Handling Outliers
R_Data['price']=np.where(R_Data['price']>=98000,R_Data['price'].median(),R_Data['price'])


# In[68]:


plot(R_Data,'price')


# In[69]:


R_Data.drop(columns=['time_taken','date'],axis=1,inplace=True)


# In[70]:


R_Data.dtypes


# In[71]:


#Applying Target Encoding on Ch_Code
ch_Code=R_Data.groupby(['ch_code'])['price'].mean().sort_values().index


# In[72]:


dict4={key:index for index,key in enumerate(ch_Code,0)}
R_Data['ch_code']=R_Data['ch_code'].map(dict4)
R_Data['ch_code']


# In[73]:


#Applying Target Encoding on Type
Type=R_Data.groupby(['type'])['price'].mean().sort_values().index
Type


# In[74]:


dict5={key:index for index,key in enumerate(Type,0)}
dict5


# In[75]:


R_Data['type']=R_Data['type'].map(dict5)
R_Data['type']


# In[76]:


#Applying Feature Selection using Mutual Info Regression
from sklearn.feature_selection import mutual_info_regression


# In[77]:


X=R_Data.drop(['price'],axis=1)
Y=R_Data['price']
X.dtypes


# In[78]:


mutual_info_regression(X,Y)


# In[79]:


#Ordering Features based on Importance
imp=pd.DataFrame(mutual_info_regression(X,Y),index=X.columns)
imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)


# In[80]:


R_Data.drop(columns=['Journey_Month','Journey_Day','stop'],axis=1,inplace=True)


# In[81]:


from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics


# spliting data into training and testing

# In[82]:


xTrain,xTest,yTrain,yTest=train_test_split(X, Y, test_size = 0.25,shuffle=True,random_state=10)


#  transforms the existing features to higher degree features.

# In[83]:


poly_features = PolynomialFeatures(degree=2)
xTrain = poly_features.fit_transform(xTrain)
xTest=poly_features.fit_transform(xTest)


# regression model

# In[84]:



reg=linear_model.LinearRegression()
Reg=reg.fit(xTrain,yTrain)


# predicting values
# 

# In[85]:


train_pred=reg.predict(xTrain)
test_pred=reg.predict(xTest)


# calculate and print MSE of training and testing
# 

# In[86]:


print('Mean Square Error', metrics.mean_squared_error(yTrain, train_pred))
print('mse of test is ',np.square(np.subtract(yTest,test_pred)).mean())
print('RMSE : ',np.sqrt(metrics.mean_squared_error(yTest,test_pred)))


# In[87]:


r2_score(yTest,test_pred)


# In[88]:


import pickle
from sklearn.metrics import accuracy_score
file=open(r'D:\ML Project\modelPolyregression.pkl','wb')
pickle.dump(Reg,file)
file.close()
forest=pickle.load(open('D:\ML Project\modelPolyregression.pkl','rb'))
ypred=forest.predict(xTest)
r2_score(yTest,ypred)


# In[ ]:




