from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import joblib



pipeline = joblib.load("/Users/sonyabudynchak/Desktop/навчання/ІТС/ргр/regression_rgr/models/random_forest_model.pkl")
data_test = pd.read_csv("/Users/sonyabudynchak/Desktop/навчання/ІТС/ргр/regression_rgr/data/test.csv")


# Розділення на ознаки та цільову змінну
X_test = data_test.drop(columns=["charges"])
y_test = data_test["charges"]

# Прогнозування на тестовому наборі
y_pred = pipeline.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)


pd.DataFrame(y_pred).to_csv('/Users/sonyabudynchak/Desktop/навчання/ІТС/ргр/regression_rgr/data/predictions.csv', index=False)

accuracy = (y_pred == y_test).mean()
print(f'Accuracy: {accuracy}')

