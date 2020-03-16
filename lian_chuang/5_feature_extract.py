
import numpy as np
import pandas as pd


class Extract:

    df = None

    def __init__(self, filename):
        df = pd.read_table(filename, header=0, sep=" ", dtype=str)
        df.drop(columns=['PassengerId', 'Name', 'Cabin',
                         'Ticket', ], axis=1, inplace=True)
        self.df = df

    def clean(self):
        self.df.replace(to_replace='female', value='-1',
                        regex=True, inplace=True)
        self.df.replace(to_replace='male', value='1', regex=True, inplace=True)
        self.df.replace(to_replace=np.nan, value='30.0',
                        regex=True, inplace=True)

    def save(self):
        self.df.to_csv(R"lian_chuang\data\myTitanic.csv",
                       index=False, header=True, na_rep="NULL")

    def oneHot(self):
        self.df['S'] = np.zeros(self.df.shape[0])
        self.df['C'] = np.zeros(self.df.shape[0])
        self.df['Q'] = np.zeros(self.df.shape[0])
        for i in range(self.df.shape[0]):
            embarked = self.df.loc[i, 'Embarked']
            self.df.loc[i, embarked] = 1
        self.df.drop(columns=['Embarked'], axis=1, inplace=True)
        self.df.drop(self.df.columns[[-1]], axis=1, inplace=True)

    def normalize(self):
        self.df = self.df[:].astype('float32')
        labels = self.df['Survived']
        self.df = (self.df-self.df.mean())/(self.df.std())
        self.df.drop(columns=['Survived'], axis=1, inplace=True)
        self.df.insert(0, 'Survived', labels)
        print(self.df)


d = Extract(R'lian_chuang\data\titanic.txt')
d.clean()
d.oneHot()
d.normalize()
d.save()
