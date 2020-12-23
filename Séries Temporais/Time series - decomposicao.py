import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('C:/Users/Pedro Fernandes/Desktop/Github/Studying/Séries Temporais/dados/AirPassengers.csv')
df.head()
# está em duas colunas, porém uma serie temporal exige que a linha do tempo esteja como índice
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('C:/Users/Pedro Fernandes/Desktop/Github/Studying/Séries Temporais/dados/AirPassengers.csv',
                 parse_dates=['Month'], index_col='Month', date_parser=dateparse)
df.index # Convertido a data com sucesso
# Criando Time Serie
ts = df['#Passengers']
ts
# Plotando ts
fig, ax=plt.subplots(figsize = (12,6))
plt.suptitle('Time Series Passengers', fontsize = 20)
plt.xlabel('Tempo', fontsize=10)
plt.ylabel('Passageiros', fontsize=10)
ax.plot(ts)
plt.show()
# Decomposição de Time Series (seazonal_decompose)
# Separação de elementos da série original
# Função do statsmodels para decomposição e subfunções: trend, seasonal, resid
decompose = seasonal_decompose(ts)

# Tendencia
tendencia = decompose.trend
fig, ax_dec = plt.subplots(figsize = (12,6))
plt.suptitle('Tendência', fontsize = 20)
plt.xlabel('Tempo', fontsize = 10)
plt.ylabel('Passageiros', fontsize = 10)
ax_dec.plot(tendencia)

# Sazonalidade
saz = decompose.seasonal
fig, ax_saz = plt.subplots(figsize = (12,6))
plt.suptitle('Sazonalidade', fontsize = 20)
plt.xlabel('Tempo', fontsize = 10)
plt.ylabel('Passageiros', fontsize = 10)
ax_saz.plot(saz)

# Aleatoriedade
aleat = decompose.resid
fig, ax_res = plt.subplots(figsize = (12,6))
plt.suptitle('Aleatoriedade', fontsize = 10)
plt.xlabel('Tempo', fontsize = 10)
plt.ylabel('Passageiros', fontsize = 10)
ax_res.plot(aleat)



