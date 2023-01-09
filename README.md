# Purchase-price-prediction_Black-Friday-Sale     EDA,FE,ML
Model to predict the purchase amount of customers against various products which will help to 
create a personalized offer for customers against different products.

# Dataset Overview
A retail company “ABC Private Limited” wants to understand the customer purchase behavior (specifically, purchase amount) 
against various products of different categories. They have shared purchase summaries of various customers for selected high-volume products from last month. These provided 
details is used as dataset for model building.

- Data set has the folloing columns
- User_ID - User ID
- Product_ID - Product ID
- Gender - Sex of User - (categorical values)
- Age - Age in bins 
- Occupation - Occupation (Masked- means alreadyconverted from categorical values to integers)
- City_Category - Category of the City (A,B,C)
- Stay_In_Current_City_Years - Number of years stay in current city
- Marital_Status - Marital Status
- Product_Category_1 - Product Category (Masked)
- Product_Category_2 - Product may belongs to other category also (Masked)
- Product_Category_3 - Product may belongs to other category also (Masked)
- Purchase - Purchase Amount (Target Variable)


# Exploratory Data Analysis (EDA)

### 1)Getting insights about the dataset

- print(df.shape)

(550068, 12)  -  This shows the dataset has 12 Columns and 550068 rows

- print(df.describe())

![Screenshot 2023-01-09 113956](https://user-images.githubusercontent.com/72013551/211259259-8de296f8-d865-45c7-9b3b-0b4f28a05fbf.png)

We can see that Count of Product_Category_3 is 166821 only means there are missing values in the columns.

- print(df.info())

![Screenshot 2023-01-09 114142](https://user-images.githubusercontent.com/72013551/211259304-4aeae9a9-9bc1-4029-bf61-0e8708b340a3.png)

### 2)Handling Missing Values

#### Check for Missing values 

 print(df.isnull().sum())
 
 ![Capture1](https://user-images.githubusercontent.com/72013551/211260652-38578495-1f9b-4acd-ab14-93f202ca1c22.PNG)

We can see from image that columns Product_Category_2 and Product_Category_3 having missing values

#### filling missing values

The columns Product_Category_2 and Product_Category_3 have the discrete values so its best to fill the missing values with mode of that column.

df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0],inplace=True)

df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0],inplace=True)

After Filling the missing values

![Capture2](https://user-images.githubusercontent.com/72013551/211261222-4d481a76-c360-42e4-a384-2e2077cfe3e1.PNG)

 Missing values successfully handeled
 
### 3)Data Visualization
#### Gender pie chart

![Screenshot 2023-01-09 114808](https://user-images.githubusercontent.com/72013551/211261779-ae599bca-043c-4a9f-8693-3618a5a916b5.png)

We can see that male customers are 2times more than female customers. so company can focus more on male related products in future.

#### Marital Status

![Screenshot 2023-01-09 114918](https://user-images.githubusercontent.com/72013551/211262230-0a50a195-f2d3-422a-986c-b445ee35cfd5.png)

single people are more than married once.

#### Age 

![Screenshot 2023-01-09 115023](https://user-images.githubusercontent.com/72013551/211262454-7e62b6af-81b5-4656-bf9d-8594970444d8.png)

Highest number of customers are of age from 26 and 30. 

#### gender vs purchase

![Screenshot 2023-01-09 115117](https://user-images.githubusercontent.com/72013551/211262863-e47d0c53-569f-4dd1-aa9d-2d189decfaeb.png)

Male customers tend to spend more than female customers

### 4)Handling Outliers

#### check for outlier with boxplot

sns.boxplot(x='Purchase', data=df)

![Screenshot 2023-01-09 120455](https://user-images.githubusercontent.com/72013551/211263371-35690a1c-8721-4506-b446-5044c8ee7611.png)

We can see from graph there are Outliers in the data

#### Removing Outliers

I have used Interquartile Range (IQR) to remove outliers.

IQR = Q3 - Q1

Q1 is defined as the middle number between the smallest number and the median of the data set.

Q3 is the middle value between the median and the highest value of the data set.

Any observations that are more than 1.5 IQR below Q1 or more than 1.5 IQR above Q3 are considered outliers.

#### After removing the outliers

![Screenshot 2023-01-09 122013](https://user-images.githubusercontent.com/72013551/211263958-6c301c44-8382-48fa-a769-76ac9e577c84.png)

# Feature Engineering

#### Find out what columns are categorical

['Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']

#### 1)Gender Column

Gender column contains  ['F' 'M']

Convert the column by Applying ordinal encoding
Map ['F' 'M'] to [ 0,1 ] using map function

#### 2)Age Column 

Age column contains bins ['0-17' '55+' '26-35' '46-50' '51-55' '36-45' '18-25']

Convert column by Applying ordinal encoding

'0-17' - 0

'18-25' - 1

'26-35' - 2

'36-45' - 3

'46-50' - 4

'51-55'  - 5

'55+' - 6

#### 3)City_Category column

City_Category column contains ['A' 'C' 'B'] 

This column contains nominal data. Use Dummy Variable Encoding using pandas.get_dummies

For Code Refer to predict.py file 

#### 4)Stay_In_Current_City_Years Column

Stay_In_Current_City_Years Column contains ['2' '4+' '3' '1' '0']

All values seen to be integer except for 4+ so replace 4+ with 4 and convert the column into integer.

#### Drop the columns that are not needed 

City_Category

Product_ID

User_ID

# Model 

#### Seprate the features and targets 

X = train_data.drop("Purchase",axis=1)  --> features

Y = train_data['Purchase']  --> targets

#### split the data for training and testing purpose

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

#### RandomForestModel

rf=RandomForestRegressor(n_estimators=150)

rf.fit(X_train, Y_train)

r_predict= rf.predict(X_test)

#### XgBoost

xg=XGBRegressor()

xg.fit(X_train, Y_train)

xg_predict= xg.predict(X_test)

#### test for the mean square error 

print(mean_absolute_error(Y_test,r_predict)) = 2216.527247585269

print(mean_absolute_error(Y_test,xg_predict)) = 2216.527247585269


Mean square error with XgBoost regressor algorithom is  less , so use the Xgboot to train on whole dataset

xg=XGBRegressor()

xg.fit(X, Y)


#### For code refer to predict.py file that i have attached in this project.





