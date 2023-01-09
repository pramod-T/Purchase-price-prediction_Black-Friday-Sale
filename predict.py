import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


#Load the dataset Csv file to dataframe

df = pd.read_csv("train.csv")


#Exploratory Data Analysis

# print(df.head(5))
# print(df.shape)
# print(df.describe())
# print(df.info())

#Deal with missing values 

#check is there any missing values
#print(df.isnull().sum())


#see what type of values in Product_Category_2
# print(df['Product_Category_2'].unique())
# print(df['Product_Category_3'].unique())

#Column nature of Product_category_2 and Product_category_3 is discret
#so filling missing values with mode is the best option.

#filling missing values with the mode and covert into integer because product_Category_1 is integer

df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0],inplace=True)
df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0],inplace=True)

# print(df.isnull().sum())
# Missing values successfully handeled


#Basic Data Visualization

#Gender
#df['Gender'].value_counts().plot(kind='pie',explode=[0.1,0],labels=['Male','Female'],title='Gender Split in data',autopct='%.1f%%')

#Marital Status
#df['Marital_Status'].value_counts().plot(kind='pie',explode=[0.1,0],labels=['Unmarried','Married'],title='Marital_Status data',autopct='%.1f%%')

#Age 
# sns.countplot(x='Age',data= df)
# plt.show()

#gender vs purchase
#sns.barplot(x='Gender',y='Purchase',data= df)
#plt.show()

#OutLiers

#check for outlier with boxplot
# sns.boxplot(x='Purchase', data=df)
# plt.show()

#using  IQR to remove outliers
Q1 = np.percentile(df['Purchase'], 25,
                interpolation = 'midpoint')
  
Q3 = np.percentile(df['Purchase'], 75,
                interpolation = 'midpoint')
IQR = Q3 - Q1
  
# print("Old Shape: ", df.shape)
  
# Upper bound
upper = np.where(df['Purchase'] >= (Q3+1.5*IQR))
  
# Lower bound
lower = np.where(df['Purchase'] <= (Q1-1.5*IQR))
  
# Removing the Outliers
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)
  
# print("New Shape: ", df.shape)
# sns.boxplot(x='Purchase', data=df)


#Feature Engineering

#Converting categorical columns into numerical
#find what are the categorical columns
list_of_cat_columns=[col for col in df.columns if df[col].dtype=='O']
print(list_of_cat_columns)

#Converting Gender column
#print(df['Gender'].unique())
#Applying ordinal encoding
df['Gender']= df['Gender'].map({'F':0,'M':1})
#print(df.head(5))


#Converting Age column
#print(df['Age'].unique())
#Applying ordinal encoding by giving integer values(0,1,2,3,4,5,6) to the ranges of age
def map_age(age):
    if age == '0-17':
        return 0
    elif age == '18-25':
        return 1
    elif age == '26-35':
        return 2
    elif age == '36-45':
        return 3
    elif age == '46-50':
        return 4
    elif age == '51-55':
        return 5
    else:
        return 6

df['Age']=df['Age'].apply(map_age)
#print(df['Age'])


#Converting City_Category Column
#print(df['City_Category'].unique())

#This column contains nominal data. Use Dummy Variable Encoding using pandas.get_dummies
# i don't have test.csv to test so i am using get_dummies, else use onehotencoding to get dummyies 

df[['City_Cat_B','City_Cat_C']]= pd.get_dummies(df['City_Category'],drop_first=True) 
#convert into int because its best to have all col in int for training
df['City_Cat_B']=df['City_Cat_B'].astype('int64')
df['City_Cat_C']=df['City_Cat_C'].astype('int64')
#print(df['City_Cat_C'])


#Converting Stay_In_Current_City_Years Column
#print(df['Stay_In_Current_City_Years'].unique())
# All seem to a number except 4+ so replace 4+ with 4 and convert col to int
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype('int64')
#print(df['Stay_In_Current_City_Years'])


#lets drop City_Category column and other unwanted coloms
df.drop('City_Category',axis=1,inplace=True)
df.drop('Product_ID',axis=1,inplace=True)
df.drop('User_ID',axis=1,inplace=True)


df= df[['Gender','Age','Occupation',
        'Stay_In_Current_City_Years','Marital_Status',
        'Product_Category_1','Product_Category_2',
        'Product_Category_3',
        'City_Cat_B','City_Cat_C','Purchase']]
#print(df.info())



#MODEl

#Split the data to train and test the model
train_data=df
#print(train_data.head())
X = train_data.drop("Purchase",axis=1)
Y = train_data['Purchase']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


#Random Forest model
# rf=RandomForestRegressor(n_estimators=150)
# rf.fit(X_train, Y_train)
# r_predict= rf.predict(X_test)

#XgBoost
xg=XGBRegressor()
xg.fit(X_train, Y_train)
xg_predict= xg.predict(X_test)

# print(mean_absolute_error(Y_test,r_predict))
print(mean_absolute_error(Y_test,xg_predict))


