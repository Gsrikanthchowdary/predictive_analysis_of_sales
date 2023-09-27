import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('sales_data.csv')
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
X_train, y_train = train_data[['feature1', 'feature2']], train_data['sales']
X_test, y_test = test_data[['feature1', 'feature2']], test_data['sales']
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')
