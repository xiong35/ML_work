
import numpy as np
import pandas as pd


class Extract:

    df = None

    def __init__(self, filename):
        df = pd.read_csv(filename, header=0, sep=",", dtype=str)
        self.df = df

    # def clean(self):
    #     self.df.replace(to_replace='female', value='-1',
    #                     regex=True, inplace=True)
    #     self.df.replace(to_replace='male', value='1', regex=True, inplace=True)
    #     self.df.replace(to_replace=np.nan, value='30.0',
    #                     regex=True, inplace=True)

    def save(self):
        self.df.to_csv(R"lian_chuang\data\myForestFire.csv",
                       index=False, header=True, na_rep="NULL")

    def oneHot(self):
        self.df.insert(0, 'weekend', np.zeros(self.df.shape[0]))
        self.df.insert(0, 'weekday', np.zeros(self.df.shape[0]))
        self.df.insert(0, 'summer', np.zeros(self.df.shape[0]))
        self.df.insert(0, 'winter', np.zeros(self.df.shape[0]))
        self.df.insert(0, 'fall', np.zeros(self.df.shape[0]))
        self.df.insert(0, 'spring', np.zeros(self.df.shape[0]))
        for i in range(self.df.shape[0]):
            if self.df.loc[i, 'month'] in ['jan', 'feb', 'mar', 'apr']:
                self.df.loc[i, 'spring'] = 1
            if self.df.loc[i, 'month'] in ['apr', 'may', 'jun', 'jul']:
                self.df.loc[i, 'summer'] = 1
            if self.df.loc[i, 'month'] in ['jul', 'aug', 'sep', 'oct']:
                self.df.loc[i, 'fall'] = 1
            if self.df.loc[i, 'month'] in ['oct', 'nov', 'dec', 'jan']:
                self.df.loc[i, 'winter'] = 1
            if self.df.loc[i, 'day'] in ['sun', 'sat']:
                self.df.loc[i, 'weekend'] = 1
            else:
                self.df.loc[i, 'weekday'] = 1
        self.df = self.df.drop(columns=['month'], axis=1)
        self.df = self.df.drop(columns=['day'], axis=1)

    def normalize(self):
        self.df = self.df[:].astype('float32')

        self.df = self.df[~self.df['area'].isin([0])]
        labels = self.df['area'].apply(lambda x: np.log(x+0.1))
        self.df = (self.df-self.df.mean())/(self.df.std())
        self.df.drop(columns=['area'], axis=1, inplace=True)
        self.df.insert(0, 'area', labels)
        print(self.df)


d = Extract(R'lian_chuang\data\forestfires.csv')
# d.clean()
d.oneHot()
d.normalize()
d.save()
