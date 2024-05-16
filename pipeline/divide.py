import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/sonyabudynchak/Desktop/навчання/ІТС/ргр/regression_rgr/data/insurance.csv')
train, test = train_test_split(df, train_size=0.8, test_size=0.2)

train.to_csv('/Users/sonyabudynchak/Desktop/навчання/ІТС/ргр/regression_rgr/data/train.csv', index=False)
test.to_csv('/Users/sonyabudynchak/Desktop/навчання/ІТС/ргр/regression_rgr/data/test.csv', index=False)