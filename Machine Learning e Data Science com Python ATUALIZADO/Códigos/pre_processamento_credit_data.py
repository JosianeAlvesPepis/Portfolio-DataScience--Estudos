# -*- coding: utf-8 -*-
# Autora: Josiane Pepis --Desenvolvido no curso ML e DT com Python

# importar a bibliotecas
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# lê o arquivo csv e adiciona ao DataFrame (df)
base = pd.read_csv("credit_data.csv")

# Mostra dados estatíscos do df
base.describe()

# count: contagem de linhas df
# mean: média de cada coluna numérica
# std: desvio padrão
# min: menor valor de cada coluna
# 25%: primeiro quartil
# 50%: mediana
# 75%: terceiro quartil
# max: valor máximo de cada coluna

# Localiza os valores que atendem a condição dentro do df
base.loc[base['age'] < 0]

# Métodos de tratamentos de dados inconsistentes

# apagar a coluna
base.drop('age', 1, inplace=True)

# apagar os registros com problema
base.drop(base[base.age <0].index, inplace=True)

# preencher os valores com a média
base['age'][base.age > 0].mean()
base.loc[base['age'] < 0, 'age'] = 40.92


# Tratamentos de dados faltantes:

#localizar valores nulos no df
pd.isnull(base['age']) #retorna todas as comparações
base.loc[pd.isnull(base['age'])] #retorna apenas as comparações verdadeiras

#Separar base para construção de ML
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Tratar valores nulos com imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0 )
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Escalonamento de atributos (implementação da técnica padronização)
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

