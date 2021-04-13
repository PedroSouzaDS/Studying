import pandas as pd
from sklearn.naive_bayes import GaussianNB # Classe do algoritmo Naive bayes
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('C:/Users/Pedro Fernandes/Desktop/Data Science/Machine Learning e Data Science com Python ATUALIZADO/Secao 4 - Aprendizagem bayesiana/risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values


# Transformar dados de texto em números para o scikit-learn
labelenc = LabelEncoder()
# Transforma coluna por coluna em numeros usando ".fit_transform"
# Em caso de grandes dimensões pode-se usar laços de repeticao
previsores[:,0] = labelenc.fit_transform(previsores[:,0])
previsores[:,1] = labelenc.fit_transform(previsores[:,1])
previsores[:,2] = labelenc.fit_transform(previsores[:,2])
previsores[:,3] = labelenc.fit_transform(previsores[:,3])

# Ativar a função que calcula a tabela de probabilidade
classificador = GaussianNB() # da inicio a tabela probabilidade
classificador.fit(previsores,classe) # diz a tabela probabilidade que é para usar esses parâmetros


# O metodo ".fit" indica que os dados estão prontos para serem submetidos ao treinamento
# treinamento do algoritmo, ordem para gerar a tabela
# Mostrar ao programa uma sequência de dados para exemplo e que ele associe o restante
resultados = classificador.predict([[0,1,0,2], [2,0,0,0], [1,0,0,2]])

# Exibindo dados da tabela probabilidade implícita
# Referente a tabela do exercício
# Os resultados vão gerando novas classificações a medida que novas combinações forem adicionadas
print(classificador.classes_) # Classes disponíveis
print(classificador.class_count_) # Quantos de cada classe existem na tabela
print(classificador.class_prior_) # Probabilidade a priori de ocorrer cada uma das classes respectivamente








