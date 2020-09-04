# Titanic dataset
# https://www.kaggle.com/startupsci/titanic-data-science-solutions

#%%
from os import remove
import pandas as pd
import numpy as np

# %%
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
#%%

# This code is written in the temporary variable style

removed_ticket_cabin = train.drop(columns=["Ticket", "Cabin"])

extract_titles = removed_ticket_cabin.assign(Title=lambda df: df.Name.str.extract(' ([A-Za-z]+)\.'), expand=False)

titles1 = extract_titles.assign(Title=lambda df: df.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'))
titles2 = titles1.assign(Title=lambda df: df.Title.replace('Mlle', 'Miss'))
titles3 = titles2.assign(Title=lambda df: df.Title.replace('Ms', 'Miss'))
titles4 = titles3.assign(Title=lambda df: df.Title.replace('Mme', 'Mrs'))

encoded_titles = titles4.assign(Title=lambda df: df.Title.map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}))
encoded_titles_final = encoded_titles.drop(columns=['Name', 'PassengerId'], axis=1)

# %%

# This code is written in the method chaining style

result = \
(
    train
    .drop(columns=["Ticket", "Cabin"])
    .assign(Title=lambda df: df.Name.str.extract(' ([A-Za-z]+)\.'), expand=False)
    .assign(Title=lambda df: df.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'))
    .assign(Title=lambda df: df.Title.replace('Mlle', 'Miss'))
    .assign(Title=lambda df: df.Title.replace('Ms', 'Miss'))
    .assign(Title=lambda df: df.Title.replace('Mme', 'Mrs'))
    .assign(Title=lambda df: df.Title.map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}))
    .drop(columns=['Name', 'PassengerId'], axis=1)
    .assign(Sex=lambda df: df.Sex.map({'female': 1, 'male': 0}).astype(int))
    .assign(Age=lambda df: infer_age(df))
    .assign(Age=lambda df: pd.cut(df.Age, 5, labels=[0,1,2,3,4]))
    .assign(Age=lambda df: df.Age.astype(int))
    # This is causing the creation of an "expand" column -- need to figure out why?
    .assign(IsAlone=lambda df: np.where(df.SibSp + df.Parch + 1 == 1, 1, 0))
    .drop(columns=['SibSp', 'Parch'])
    .assign(Pclass=lambda df: df.Pclass.astype(int))
    .assign(AgeClass=lambda df: df.Age * df.Pclass)
    .assign(Embarked=lambda df: df.Embarked.fillna(df.Embarked.dropna().mode()[0]))
    .assign(Embarked=lambda df: df.Embarked.map({'S': 0, 'C': 1, 'Q': 2}))
    .assign(Fare=lambda df: df.Fare.fillna(df.Fare.dropna().median()))
    .assign(Fare=lambda df: pd.qcut(df.Fare, 4, labels=[0,1,2,3]))
    .drop(columns=['expand'])
)
result

#%%
# This data cleaning task is interesting ... it requires us to infer age for missing values based
# on passenger class x gender correlations

def infer_age(df):
    for sex in range(0, 2):
        for pclass in range(1, 4):
            age_guess = df[(df.Sex == sex) & (df.Pclass == pclass)].Age.dropna().median()
            age_guess = int(age_guess/0.5 + 0.5) * 0.5
            df.loc[(df.Age.isnull()) & (df.Sex == sex) & (df.Pclass == pclass), 'Age'] = age_guess
    return df.assign(Age=lambda df: df.Age.astype(int)).Age

#%%
# Now let's convert ages into age bands by binning into 5 equal-sized bands

pd.cut(result.Age, 5, labels=[0,1,2,3,4])
#%%
#%%
# This is an interesting expression that lets us see whether we got the right result or not
# So a visualization could be one where the user wants to see this as the output
result[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# %%
pd.crosstab(result.Title, result.Sex)