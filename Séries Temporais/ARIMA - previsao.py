import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA # Habilita funcao ARIMA
from pmdarima.arima import auto_arima # Habilita funcao AUTO ARIMA
from datetime import datetime

# Importa dados
df = pd.read_csv('C:/Users/Pedro Fernandes/Desktop/Data Science/Séries Temporais/dados/AirPassengers.csv')

dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('C:/Users/Pedro Fernandes/Desktop/Data Science/Séries Temporais/dados/AirPassengers.csv',
                 parse_dates=['Month'], index_col='Month', date_parser=dateparse)
df.head()

# Define Time Serie ts
ts = df['#Passengers']
fig, ax = plt.subplots(figsize = (12,6))
plt.suptitle('Time Series Passengers', fontsize = 20)
plt.xlabel('Time', fontsize = 10)
plt.ylabel('Flights', fontsize = 10)
ax.plot(ts)

# Criando modelo ARIMA parametros p=2, q=1, d=2. Treinamento e visualização dos resultados
modelo = ARIMA(ts, order=(2,1,2), freq=ts.index.inferred_freq)
modelo_train = modelo.fit()
modelo_train.summary() 

# Previsões para o futuro
previsoes = modelo_train.forecast(steps = 12)[0]
previsoes # equivale a "next12"
# PREVISAO COM ARIMA MANUAL
# Serie Temporal completa com previsões e eixo
# diferente do curso, fazer eixos personalizados com matplotlib
# este modelo contem visualizacao personalizada SEM O ARIMA AINDA
fig, ax1 = plt.subplots(figsize = (12,6))
plt.suptitle('Previsão', fontsize = 20)
plt.xlabel('Meses', fontsize = 10)
plt.ylabel('Passageiros', fontsize = 10)
ax1.plot(ts) # a variavel ax1 assume a forma da serie temporal que adiante é usada 
             # na previsão do ARIMA junto a valiavel 'modelo_train'
# Serie com a previsao ARIMA
modelo_train.plot_predict('1960-01-01', '1965-01-01', ax=ax1, plot_insample=True)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==
# OBS: A previsao numerica e a previsao do grafico sao independentes...
# ...para fazer o grafico o programa calcula a parte numerica automaticamente
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==

# PREVISAO COM AUTO ARIMA
# Escolhe o melhor modelo, com menores indices automaticamente
# Outro processo diferente e independente do ARIMA manual
AutoArima = auto_arima(ts, m=12, seasonal=True, trace=False)
AutoArima.summary()
# Previsao desejada para os próximos 12 meses
next12 = AutoArima.predict(n_periods = 12)
next12 # equivale a "previsoes"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==
# Não Há como extrair visualizações do AUTO ARIMA, a solucao é...
# ... gerar apenas o relatório do AUTO ARIMA e caso queira uma visualização jogar...
# ...os dados do relatório gerado no método do ARIMA manual
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==