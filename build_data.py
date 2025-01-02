import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df, test_df):
    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    imputer_age = SimpleImputer(strategy='median')
    train_df['Age'] = imputer_age.fit_transform(train_df[['Age']])
    test_df['Age'] = imputer_age.transform(test_df[['Age']])

    imputer_embarked = SimpleImputer(strategy='most_frequent')
    train_df['Embarked'] = imputer_embarked.fit_transform(train_df[['Embarked']]).ravel()

    imputer_fare = SimpleImputer(strategy='median')
    test_df['Fare'] = imputer_fare.fit_transform(test_df[['Fare']])

    le = LabelEncoder()
    train_df['Sex'] = le.fit_transform(train_df['Sex'])
    test_df['Sex'] = le.transform(test_df['Sex'])

    train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        if col != 'Survived':
            test_df[col] = 0

    test_df = test_df[train_df.drop('Survived', axis=1).columns]

    scaler = StandardScaler()
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
    train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])

    return train_df, test_df


def split_data(train_df, test_size=0.2, random_state=42):
    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_val, y_train, y_val


def get_processed_data(train_path='train.csv', test_path='test.csv'):
    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df = preprocess_data(train_df, test_df)
    X_train, X_val, y_train, y_val = split_data(train_df)
    return X_train, X_val, y_train, y_val, test_df
