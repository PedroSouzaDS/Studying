import pandas as pd

base = pd.read_csv('C:/Users/Pedro Fernandes/Desktop/Github/Studying/Machine Learning/Machine Learning e Data Science com Python ATUALIZADO/Secao 3 - Pre-processamento com Pandas e scikit-learm/credit_data.csv')
base.columns = ['id', 'income', 'age', 'loan', 'default']
base.describe()

# VALORES NULOS / FALTANTES

# Usa-se info para ver quantidades de valores nulos e tipos de variaveis
# com ele identificamos que ha dados nulos em 'age'
base.info()
# Identificando dados vazios em 'age'
pd.isnull(base['age']) # classifica todos os dados como true ou false
base.loc[pd.isnull(base['age'])]

# Substituindo valores vazios pela média de duas formas
 # 1 - pd.loc
base['age'][base['age'] >= 0]
media = base['age'][base['age'] >= 0].mean()
base.loc[base['age'].isnull(), 'age'] = media
base.info()


 # 2 - pd.fillna
 
base.fillna(base['age'].mean())
base = base.fillna(base['age'].mean())
base.info()

# 3 - imputer Scikit Learn

from sklearn.impute import SimpleImputer
import numpy as np

# Cria classificadores e previsores
previsores = base.iloc[:, 1:4].values # id nao entra
classificadores = base.iloc[:, 4].values # values indica ser uma matriz numerica

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# fit: encaixa os novos valores nos locais desejados
imputer = imputer.fit(previsores[:, 0:3])
# transform: depois de encaixados os valores 'nan' são transformados em medias
previsores = imputer.transform(previsores[:, 0:3])
# Confere se os valores nulos foram preenchidos
base.info()


                
