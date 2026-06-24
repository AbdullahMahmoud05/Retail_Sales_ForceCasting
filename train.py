#Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

df = pd.read_csv('retail_store_sales_cleaned.csv')

#features 2D
x = df.drop(['Item','Total Spent','Transaction Date'], axis=1)

#targrt 2D
y=df['Total Spent']

# Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Model
model = LinearRegression()

#Fit
model.fit(X_train,y_train)

#Predict
y_pred = model.predict(X_test)



print("R² =", r2_score(y_test, y_pred))

print("Mean Squared Error =", mean_squared_error(y_test, y_pred))

print("Mean Absolute Error =", mean_absolute_error(y_test, y_pred))

# Save model
joblib.dump(model, "sales_model.pkl")

# Save columns
joblib.dump(x.columns.tolist(), "model_columns.pkl")