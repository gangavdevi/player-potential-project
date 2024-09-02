

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# !pip install category_encoders
# import category_encoders as ce
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Load the data
data = pd.read_csv("C:/Users/ganga/Downloads/OneDrive/Desktop/player_ potential_flask/football player dataset.csv", encoding='latin-1')
# data.head()

# data.info()

# data.columns

# data.describe()

# data.shape

# data.isnull().sum()

# data['Special'].unique()

# data['Joined'].unique()

# data['Contract Valid Until'].unique()

# data['Loaned From'].unique()

#Data cleaning
#dropping unnecessary columns as they are likely irrelevant to the modeling process
data.drop(columns=['Unnamed: 0','ID', 'Name', 'Photo','Nationality', 'Flag', 'Club Logo','Special', 'Real Face', 'Jersey Number', 'Joined', 'Loaned From',
  'Contract Valid Until'],inplace=True)

# data.info()

# data.duplicated().sum()

# data.shape

# data['Release Clause'].unique()

def clean_currency(x):
    if isinstance(x, str):
        # Remove non-numeric characters
        x = ''.join(c for c in x if c.isdigit() or c in ['.', '€', 'M'])
        return float(x.replace('€', '').replace('M', '')) * 1000000  # Convert millions to actual numbers
    return x

data['Release Clause'] = data['Release Clause'].apply(clean_currency)
data['Release Clause'] = pd.to_numeric(data['Release Clause'], errors='coerce')  # Handle potential errors

# data['Release Clause'].dtype

# data['Release Clause'].value_counts()

# data['Value'].unique()

# data['Wage'].unique()

def convert_value_wage(value):
    if isinstance(value, str):
        try:
            # Remove any invalid characters
            value = value.encode("ascii", "ignore").decode()
            # Handle different suffixes
            if 'M' in value:
                return float(value.replace('M', '').replace('€', '').replace('K', '')) * 1e6
            elif 'K' in value:
                return float(value.replace('K', '').replace('€', '')) * 1e3
            else:
                return float(value.replace('€', ''))
        except ValueError:
            # If conversion fails, return NaN or some default value
            return np.nan
    return value

# Apply the function to the 'Value' and 'Wage' columns
data['Value'] = data['Value'].apply(convert_value_wage)
data['Wage'] = data['Wage'].apply(convert_value_wage)

# data['Value'].value_counts()

# data['Value'].dtype, data['Wage'].dtype

# Function to convert height from different formats to centimeters
def convert_height(height_str):
    if isinstance(height_str, str):  # Check if it's a string
        if "cm" in height_str:
            return float(height_str.replace("cm", ""))
        elif "'" in height_str:
            feet, inches = map(int, height_str.replace('"', '').split("'"))
            total_inches = (feet * 12) + inches
            return total_inches * 2.54
    return np.nan  # Return NaN if it's not a string or invalid

# Convert 'Height' and 'Weight' columns to numeric
data['Height'] = data['Height'].apply(convert_height)

# Convert 'Weight' from 'lbs' to 'kg'
data['Weight'] = data['Weight'].replace({'lbs': ''}, regex=True).astype(float) * 0.453592

# Checking first few rows to ensure conversion was successful
print(data[['Height', 'Weight']].head())

# data['Height'].dtype, data['Weight'].dtype

# data['Height'].value_counts()

def convert_position_value_to_mean(position_str):
    # Check if the value is a string, otherwise return NaN
    if isinstance(position_str, str):
        # Split the string by '+' and convert the parts to integers
        values = list(map(int, position_str.split('+')))
        # Return the mean of the values
        return np.mean(values)
    else:
        return np.nan

# Apply the conversion function to all position columns
position_columns = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
                    'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM',
                    'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB',
                    'RCB', 'RB']

for col in position_columns:
    if data[col].dtype == object:  # Check if column contains strings
        data[col] = data[col].apply(convert_position_value_to_mean)
    else:
        print(f"Skipping column '{col}' as it does not contain strings.")

# data.info()



# data['Body Type'].value_counts()

#replacing error values of'Body Type' with mode 'Normal'
data['Body Type'] = data['Body Type'].replace(['Messi','C. Ronaldo','Neymar','Courtois','PLAYER_BODY_TYPE_25','Shaqiri','Akinfenwa'],'Normal')
# data['Body Type'].value_counts()

# histplot=data.hist(figsize=(20,15))

# data.describe()

# data.columns



#handling missing values using mean for normal distribution
# for i in ['Age', 'Overall', 'Potential','Reactions','Jumping']:
#   data[i].fillna(data[i].mean(),inplace=True)

# #handling missing values using median for skewed data
# for i in ['Value','International Reputation', 'Weak Foot', 'Skill Moves', 'Height', 'Weight', 'LS', 'ST', 'RS', 'LW',
#        'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
#        'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB',
#        'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
#        'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
#        'Acceleration', 'SprintSpeed', 'Agility', 'Balance',
#        'ShotPower', 'Stamina', 'Strength', 'LongShots',
#        'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
#        'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
#        'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes','Release Clause']:
#   data[i].fillna(data[i].median(),inplace=True)


# Handling missing values using mean for normally distributed data
for i in ['Age', 'Overall', 'Potential','Reactions','Jumping']:
    data[i] = data[i].fillna(data[i].mean())

# Handling missing values using median for skewed data
for i in ['Value','Wage','International Reputation', 'Weak Foot', 'Skill Moves', 'Height', 'Weight', 'LS', 'ST', 'RS', 'LW',
          'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
          'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB',
          'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
          'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
          'Acceleration', 'SprintSpeed', 'Agility', 'Balance',
          'ShotPower', 'Stamina', 'Strength', 'LongShots',
          'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
          'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
          'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes','Release Clause']:
    data[i] = data[i].fillna(data[i].median())


# data.isnull().sum()

# data['Club'].value_counts()

# data['Preferred Foot'].value_counts()

# data['Work Rate'].value_counts()

# data['Body Type'].value_counts()

# data['Position'].value_counts()

# Drop columns with more than 50% missing values
data.dropna(thresh=0.5 * len(data), axis=1, inplace=True)

# Handle missing values for categorical columns by replacing with mode
# data['Club'].fillna(data['Club'].mode()[0], inplace=True)
# data['Preferred Foot'].fillna(data['Preferred Foot'].mode()[0], inplace=True)
# data['Work Rate'].fillna(data['Work Rate'].mode()[0], inplace=True)
# data['Body Type'].fillna(data['Body Type'].mode()[0], inplace=True)
# data['Position'].fillna(data['Position'].mode()[0], inplace=True)

data['Club'] = data['Club'].fillna(data['Club'].mode()[0])
data['Preferred Foot'] = data['Preferred Foot'].fillna(data['Preferred Foot'].mode()[0])
data['Work Rate'] = data['Work Rate'].fillna(data['Work Rate'].mode()[0])
data['Body Type'] = data['Body Type'].fillna(data['Body Type'].mode()[0])
data['Position'] = data['Position'].fillna(data['Position'].mode()[0])


# data.isna().sum()

# Feature Engineering
data['Offensive_Score'] = data[['Finishing', 'ShotPower', 'LongShots', 'Positioning']].mean(axis=1)
data['Defensive_Score'] = data[['Marking', 'StandingTackle', 'SlidingTackle', 'Interceptions']].mean(axis=1)
data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
data['Pace'] = data[['Acceleration', 'SprintSpeed']].mean(axis=1)
data['Technical_Skill'] = data[['Dribbling', 'BallControl', 'Agility']].mean(axis=1)

#mean provides a balanced average of the continuous performance metrics, reflecting the overall average from each attribute.

# Positional Strengths calculating
def calculate_position_strength(data,positions):
    return data[positions].mean(axis=1)
#defining positions
forward_positions = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW']
data['Forward_Strength'] = calculate_position_strength(data,forward_positions)

midfielder_positions = ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB']
data['Midfielder_Strength'] = calculate_position_strength(data,midfielder_positions)

defender_positions = ['LB', 'LCB', 'CB', 'RCB', 'RB']
data['Defender_Strength'] = calculate_position_strength(data,defender_positions)

goalkeeper_positions = ['GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
data['Goalkeeper_Strength'] = data[goalkeeper_positions].mean(axis=1)

# data.shape

# data.columns

data.drop(columns=['Finishing', 'ShotPower', 'LongShots', 'Positioning','Marking', 'StandingTackle', 'SlidingTackle', 'Interceptions','Height', 'Weight',
'Acceleration', 'SprintSpeed','Dribbling', 'BallControl', 'Agility','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM','LWB','LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB','RCB', 'RB',
'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'],inplace=True)

# data.columns

# Create pairplots for selected variables to visualize their relationships

# # Visualize relationship between 'Overall' and 'Potential'
# sns.lmplot(x='Overall', y='Potential', data=data)
# plt.title('Relationship Between Overall and Potential')
# plt.show()

# # Visualize relationship between 'Age' and 'Potential'
# sns.lmplot(x='Age', y='Potential', data=data)
# plt.title('Relationship Between Age and Potential')
# plt.show()

# data.shape



# #boxplot to check for outliers
# numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
# num_rows = (len(numerical_columns) + 5) // 6  # Round up to the nearest integer

# plt.figure(figsize=(20, 5 * num_rows))  # Adjust figure height based on number of rows
# for i, col in enumerate(numerical_columns):
#     plt.subplot(num_rows, 6, i+1)  # Dynamically adjust number of rows
#     sns.boxplot(y=data[col])
#     plt.title(f"Boxplot of {col}")

# plt.tight_layout()
# plt.show()



#check the range of weak foot column, normal range is 3-5 star rating
# data['Weak Foot'].max(),data['Weak Foot'].min()

# #check the range of shortpassing column, normal range ranking is 35-95
# data['ShortPassing'].max(),data['ShortPassing'].min()

# #check the range of reaction column
# data['Reactions'].max(),data['Reactions'].min()

# data['Balance'].max(),data['Balance'].min()

# data['Jumping'].max(),data['Jumping'].min()

# data['Goalkeeper_Strength'].max(),data['Goalkeeper_Strength'].min()

# data['BMI'].max(),data['BMI'].min()

# data['Overall'].max(),data['Overall'].min()

# data['Potential'].max(),data['Potential'].min()

# data['Value'].max(),data['Value'].min()#need to handle

# data['International Reputation'].max(),data['International Reputation'].min()

# data['HeadingAccuracy'].max(),data['HeadingAccuracy'].min()

# data['Penalties'].max(),data['Penalties'].min()

# data['Release Clause'].max(),data['Release Clause'].min()

#plot Value against index
# plt.figure(figsize=(10, 6))
# plt.plot(data.index, data['Value'], marker='o', linestyle='-', color='b')
# plt.title('Value vs Index')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)
# plt.show()

#handle the outlier of value
data['Value_log'] = np.log1p(data['Value'])



#check the unique values of the categorical columns
# for col in data.select_dtypes(include=['object']):
#   print(col)
#   print(data[col].nunique())

# #valuecount of club
# data['Club'].value_counts()

# data['Preferred Foot'].value_counts()

#encoding
data_encoded = data.copy()

# Initialize the Label Encoder
label_encoder = LabelEncoder()

# Apply Label Encoding to the target variable
data_encoded['Potential'] = label_encoder.fit_transform(data_encoded['Potential'])

categorical_columns = ['Club', 'Preferred Foot', 'Work Rate', 'Body Type', 'Position']

#apply frequency encoding

for col in categorical_columns:
    frequency_map=data_encoded[col].value_counts(normalize=True)

    data_encoded[col]=data_encoded[col].map(frequency_map)

data_encoded

# # Calculate correlation matrix
# corr_matrix = data_encoded.corr()
# corr_matrix

# Plotting the heatmap
# plt.figure(figsize=(45,25))
# sns.heatmap(corr_matrix, annot=True)
# plt.title('Correlation Heatmap')
# plt.show()

relevancy_check = ['Release Clause', 'Potential']
data_relevant = data[relevancy_check].copy()

# correlation_matrix = data_relevant.corr()

# plt.figure(figsize=(6, 4))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix of Release clause relevancy')
# plt.show()

#drop club
data_encoded.drop(columns=['Club'],inplace=True)

# data_encoded.columns

# select features and target
x=data_encoded.drop(columns=['Potential'])
y=data_encoded['Potential']

# x.columns

# y

#model identification
# splitting, split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# x_train.shape

# y_train.shape

# x_test.shape

# y_test.shape

# Initialize the RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
#training the model random forest
rfr.fit(x_train,y_train)
# Calculate feature importances
feature_importances = rfr.feature_importances_

# Ensuring the number of features matches the length of feature_importances
relevant_features = x_train.columns[:len(feature_importances)]  

# Create a DataFrame to store feature names and their corresponding importances
feature_importance_df = pd.DataFrame({'Feature':relevant_features, 'Importance': feature_importances})
#feature_importance_df
# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()

# Initialize RFE with the random forest regressor and select 20 features
rfe = RFE(estimator=rfr, n_features_to_select=20)
# Fit RFE on the training data
x_train_rfe = rfe.fit_transform(x_train, y_train)
# Transform the test data
x_test_rfe = rfe.transform(x_test)

# printing the selected 20 features, exclude the target column potential using RFE
features_name=data_encoded.drop(columns=['Potential']).columns[rfe.support_]
features_name

#scale the selected features
sc_rfe=StandardScaler()
x_train_rfe_sc=sc_rfe.fit_transform(x_train_rfe)
x_test_rfe_sc=sc_rfe.transform(x_test_rfe)

# x_train_rfe_sc

# x_test_rfe_sc

#training the model random forest
rfr.fit(x_train_rfe_sc,y_train)

#predicting the model
y_pred=rfr.predict(x_test_rfe_sc)
y_pred

#evaluating using mean squared error ,model evaluation
mse=mean_squared_error(y_test,y_pred)
mse

#mean absolute error
mae=mean_absolute_error(y_test,y_pred)
mae

#r2 score
r2=r2_score(y_test,y_pred)
r2

#traing the model using linear regression
# lr=LinearRegression()
# lr.fit(x_train_rfe_sc,y_train)

# #predicting
# y_pred_lr=lr.predict(x_test_rfe_sc)
# y_pred_lr

# #evaluating
# mse_lr=mean_squared_error(y_test,y_pred_lr)
# mse_lr

# #mean absolute error
# mae_lr=mean_absolute_error(y_test,y_pred_lr)
# mae_lr

# #r2 score
# # r2_lr=r2_score(y_test,y_pred_lr)
# # r2_lr

# # Training the model using SVR
# svr = SVR()
# svr.fit(x_train_rfe_sc, y_train)

# #predicting
# y_pred_svr = svr.predict(x_test_rfe_sc)
# y_pred_svr

# #evaluating
# mse_svr = mean_squared_error(y_test, y_pred_svr)
# mse_svr

# #mean absolute error
# mae_svr = mean_absolute_error(y_test, y_pred_svr)
# mae_svr

# #r2
# r2_svr = r2_score(y_test, y_pred_svr)
# r2_svr

# #training  the model using knn regressor
# knn = KNeighborsRegressor()
# knn.fit(x_train_rfe_sc, y_train)

# #predicting
# y_pred_knn = knn.predict(x_test_rfe_sc)
# y_pred_knn

# #evaluating
# #mean squared error
# mse_knn = mean_squared_error(y_test, y_pred_knn)
# mse_knn

# #mean absolute error
# mae_knn = mean_absolute_error(y_test, y_pred_knn)
# mae_knn

# #r2_score
# r2_knn = r2_score(y_test, y_pred_knn)
# r2_knn

# #training the model using decision tree regressor
# dt = DecisionTreeRegressor()
# dt.fit(x_train_rfe_sc, y_train)

# #predicting
# y_pred_dt = dt.predict(x_test_rfe_sc)
# y_pred_dt

# #evaluating
# #mean squared error
# mse_dt = mean_squared_error(y_test, y_pred_dt)
# mse_dt

# #mae
# mae_dt = mean_absolute_error(y_test, y_pred_dt)
# mae_dt

# #r2
# r2_dt = r2_score(y_test, y_pred_dt)
# r2_dt

#hyper parameter tuning rfr
# param_grid_rf = {'n_estimators': [50, 100, 200],'max_depth': [None, 10, 20],'min_samples_split': [2, 5, 10]}

# #GridSearchCV on trained model
# grid_rf = GridSearchCV(rfr, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
# grid_rf.fit(x_train_rfe_sc, y_train)

# #getting best parameter
# best_rf = grid_rf.best_estimator_
# best_rf

# #predict using best model
# y_pred_grid_rf = best_rf.predict(x_test_rfe_sc)
# y_pred_grid_rf

# #evaluate
# mse_grid_rf = mean_squared_error(y_test, y_pred_grid_rf)
# mse_grid_rf

# #mae
# mae_grid_rf = mean_absolute_error(y_test, y_pred_grid_rf)
# mae_grid_rf

#r2
# r2_grid_rf = r2_score(y_test, y_pred_grid_rf)
# r2_grid_rf

# Function to predict player potential based on a new set of features
# def predict_player_potential(features, feature_names):  # Updated parameter name to 'feature_names'
#     # Print each feature name along with its corresponding input value
#     for feature_name, feature_value in zip(feature_names, features):
#         print(f"{feature_name}: {feature_value}")

#     # Reshape the input features to match the model's expected input shape
#     features = np.array(features).reshape(1, -1)

#     # Scale the features based on the fitted scaler (sc_rfe)
#     features_scaled = sc_rfe.transform(features)  # Ensure sc_rfe has been fitted before this step

    # Predict the outcome using the trained RandomForestRegressor model (rfr)
#     prediction = rfr.predict(features_scaled)

#     # Return the predicted potential value
#     return prediction[0]

# #feature values corresponding to the selected features
# feature_values = [31.0,94.0,110500000.0,84.0,86.0,93.0,87.0,68.0,75.0,96.0,226500000.0,92.00,27.25,24.902645,88.5,94.666667,46.4375,26.9,10.8,18.520526] #feature values for a player
# feature_names = ['Age', 'Overall', 'Value', 'Crossing', 'Volleys', 'Curve',
 #      'LongPassing', 'Jumping', 'Penalties', 'Composure', 'Release Clause',
  #     'Offensive_Score', 'Defensive_Score', 'BMI', 'Pace', 'Technical_Skill',
   #    'Forward_Strength', 'Defender_Strength', 'Goalkeeper_Strength',
    #   'Value_log']  # feature names

# # Predict and print the player's future potential
# predicted_potential = predict_player_potential(feature_values, feature_names)
# print(f"The future potential of a player is: {predicted_potential}")

# pd.set_option('display.max_columns', None)
# df=pd.DataFrame(data_encoded)
# df.head()




import pickle
pickle.dump(rfr,open('model_out.pkl',"wb"))



pickle.dump(sc_rfe,open('scaler.pkl',"wb"))


