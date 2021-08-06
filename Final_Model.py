import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('test.csv')


def preprocess(dataset):
  dataset = dataset.drop(columns = ['product-type']) 
  
  # Valores Faltantes
  dataset.replace('?', np.nan, inplace=True)
  dataset['formability'] = dataset['formability'].astype(float)
  dataset.isnull().sum()

  #Carregando as colunas trabalhadas e eliminando as não usadas
  colunas_faltantes = pickle.load(open('colunas_faltantes.txt', 'rb'))
  dataset = dataset.drop(columns = colunas_faltantes)

  # Foi observado um padrão forte entre a classe 'Ideal' com a coluna 'Temper_rolling'
  dataset.loc[dataset['temper_rolling'].isnull(), 'temper_rolling'] = 'NULA'
  dataset.loc[dataset['bf'].isnull(), 'bf'] = 'NULA'
  dataset.loc[dataset['bl'].isnull(), 'bl'] = 'NULA'
  dataset.loc[dataset['oil'].isnull(), 'oil'] = 'NULA'
  dataset.loc[dataset['cbond'].isnull(), 'cbond'] = 'NULA'
  dataset.loc[dataset['bt'].isnull(), 'bt'] = 'NULA'

  #----Substituição por Padrão Encontrado
  dataset.loc[dataset['surface-quality'] == 'E', 'condition'] = 'S' 
  dataset.loc[dataset['surface-quality'] == 'D', 'condition'] = 'S' 
  dataset.loc[dataset['surface-quality'] == 'F', 'condition'] = 'S'
  dataset.loc[dataset['condition'] == 'A', 'surface-quality'] = 'G' 
  dataset.loc[dataset['steel'] == 'A', 'condition'] = 'S' 
  dataset.loc[dataset['steel'] == 'M', 'surface-quality'] = 'G' 
  dataset.loc[(dataset['surface-quality'] == 'G') & (dataset['condition']=='A'), 'steel'] = 'R'
  dataset.loc[(dataset['steel'] == 'R') & (dataset['condition'] =='S'), 'surface-quality'] = 'E'
  dataset.loc[dataset['family'] == 'TN', 'steel'] = 'A'

  return dataset


def random_input(df, coluna):
    numeros_faltantes = df[coluna].isnull().sum()
    valores_observados = df.loc[df[coluna].notnull(),coluna]
    df.loc[df[coluna].isnull(), coluna + '_imputation'] = np.random.choice(valores_observados, 
                                                                           numeros_faltantes, replace = True)
    return df


def encoding(dataset): 
  label_encoder = LabelEncoder()
  for coluna in colunas_faltantes:
      encoded = dataset.iloc[:, dataset.columns.get_loc(coluna)].values
      dataset[coluna + '_imputation'] = dataset[coluna]
      dataset = random_input(dataset,coluna)
      if coluna != 'formability':
          encoded=label_encoder.fit_transform(dataset[coluna 
                                                      + '_imputation'].values)
          dataset[coluna + 
                  '_imputation'] = dataset[coluna + 
                                          '_imputation'].replace(dataset[coluna + 
                                                                        '_imputation'].values.tolist(),encoded)

  for coluna in dataset.columns.values:
      if colunas_faltantes.count(coluna) > 0:
          encoded = dataset.iloc[:,dataset.columns.get_loc(coluna + '_imputation')].values
          dataset[coluna] = dataset[coluna].replace(dataset[coluna].values.tolist(), encoded)
      if type(dataset[coluna][0]) == str or coluna == 'family':
          encoded = label_encoder.fit_transform(dataset[coluna].values)
          dataset[coluna] = dataset[coluna].replace(dataset[coluna].values.tolist(), encoded)

  deter_data = pd.DataFrame(columns = ['Det' + coluna for coluna in colunas_faltantes])
  for coluna in colunas_faltantes:
      deter_data['Det' + coluna] = dataset[coluna + '_imputation']
      parameters = list(set(dataset.columns) - set(colunas_faltantes)-{coluna + '_imputation'} - {'recozimento'})

      model = LinearRegression()
      model.fit(X = dataset[parameters], y = dataset[coluna + '_imputation'])
      
      deter_data.loc[dataset[coluna].isnull(), 'Det' + coluna] = model.predict(dataset[parameters])[dataset[coluna].isnull()]
      dataset[coluna] = dataset[coluna + '_imputation']
      dataset = dataset.drop(columns = coluna + '_imputation')


  return dataset      
        
dataset = preprocess(dataset)

#---Imputação de valores por Regressão Linear
colunas_faltantes = []
for i in range(len(dataset.isnull().sum())):
    if dataset.isnull().sum()[i] > 1:
        colunas_faltantes.append(dataset.isnull().sum().index[i])

dataset = encoding(dataset)
  

x_data = dataset.iloc[:, 0:dataset.shape[1] - 1].values
ids = dataset.iloc[:, dataset.shape[1] - 1].values

    ##--Escalonamento dos dados
from sklearn.preprocessing import StandardScaler 
scaler_data = StandardScaler()
x_data = scaler_data.fit_transform(x_data)


#---- Importação dos Algoritmos

random = pickle.load(open('random_finalizado.sav', 'rb'))
knn = pickle.load(open('knn_finalizado.sav', 'rb'))
svm = pickle.load(open('svm_finalizado.sav', 'rb'))

#---- Outputs dos Algoritmos
previsoes_random = random.predict(x_data)
previsoes_knn = knn.predict(x_data)
previsoes_svm = svm.predict(x_data)

#----- Decisão Final
prev_final = []
for i in range(x_data.shape[0]):
    if previsoes_random[i] == 1:
        prev_final.append(1)
    elif previsoes_knn[i] == 2:
        prev_final.append(2)
    elif previsoes_random[i] == 0:
        prev_final.append(0)
    else:
        prev_final.append(previsoes_svm[i])
        
prev_final2 = []
for i in prev_final:
    if i == 2:
        prev_final2.append('ruim')
    elif i == 1:
        prev_final2.append('mediano')
    else:
        prev_final2.append('ideal')


#--- Criação do Arquivo CSV Final
dataset = np.column_stack((ids,prev_final2))
previsao_final = pd.DataFrame(dataset,columns=['id','recozimento'])
previsao_final.to_csv(r'previsao_final.csv', index = False)

