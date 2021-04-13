import pandas as pd
import numpy as np

base = pd.read_csv('C:/Users/Pedro Fernandes/Desktop/Machine Learning e Data Science com Python ATUALIZADO/Secao 3 - Pre-processamento com Pandas e scikit-learm/credit_data.csv')
base.columns = ['id', 'income', 'age', 'loan', 'default']

base.describe()
# corrigindo valores inconsistentes
# idade negativa
base['age'][base['age'] < 0]
base.loc[base['age'] < 0]
# Substituindo esses valores negativos pela média dos valores positivos
base['age'][base['age'] >= 0]
media = base['age'][base['age'] >= 0].mean()
base.loc[base['age'] < 0, 'age'] = media
# Averiguando a substituição
base.iloc[[15, 21, 26], :]

# corrigindo VALORES FALTANTES
base.info()
# Valores faltantes em 'age'

# 1a opcao para a substituicao
base.loc[base['age'].isnull(), 'age'] = media
base.iloc[[28, 31, 32], :]
base.info()

# 2a opcao: sklearn
# separa classificadores de previsores
# previsores = base.iloc[:, 1:4].values 
# classificadores = base.iloc[:, 4].values

# garantir que valores 'nan' sejam 'np.nan'

# base.loc[base['age'].isnull(), 'age']

#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values='NaN', strategy='mean')
#imputer = imputer.fit(previsores[:, 0:3])
#previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
#base.info()

# CONTINUA COM ESCALONAMENTO

