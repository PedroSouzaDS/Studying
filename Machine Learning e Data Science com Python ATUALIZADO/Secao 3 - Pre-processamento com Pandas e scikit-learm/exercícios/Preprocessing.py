import pandas as pd

base = pd.read_csv('C:/Users/Pedro Fernandes/Desktop/Github/Studying/Machine Learning/Machine Learning e Data Science com Python ATUALIZADO/Secao 3 - Pre-processamento com Pandas e scikit-learm/credit_data.csv')
base.columns = ['id', 'income', 'age', 'loan', 'default']
base.describe()

# VALORES INCONSISTENTES

# percebe-se idades negativas na coluna 'age'
# Usar comandos para encontrar a idade negativa
base.describe()
base.loc[base['age'] < 0]
# Trata-se de três abordagens
# 1 - Apagar coluna
base.drop('age', axis = 1, inplace = True)
# 2 - Apagar linhas
base.drop(base[base['age'] < 0].index, axis = 0, inplace = True)
# 3 - Preencher manualmente
 # 3.1 - Substituir 'na mão' consultando a idade das pessoas
 # 3.2 - Substituir pela média das idades (mais viável)
 # IMPORTANTE: Analisar variância para 
base.mean() # Média errada, contando com os valores negativos
base['age'].mean() # Média errada, contando com os valores negativos
media = base['age'][base['age'] >= 0].mean() # Média das idades positivas
base.loc[base['age'] < 0, 'age'] = media
base.loc[[15, 21, 26], :]

