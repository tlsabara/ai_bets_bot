#%%
# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import keras
# Pandas é claro!!
# sklearn não podia faltar, train_test_split é para a divisão da base.
# Keras que é o cara novo, borave.

#%%
# Variáveis

# Realizei o carregamento dos arquivos.

previsores_file = './datasheets/entradas_breast.csv'
df_previsores = pd.read_csv(previsores_file)

class_file = './datasheets/saidas_breast.csv'
df_class_file = pd.read_csv(class_file)

#%%
# Jutsus

# Separação nas bases de treinamento e de teste
df_prev_training, df_prev_test, df_class_training, df_class_test = train_test_split(
    df_previsores,
    df_class_file,
    test_size=0.25
)

# Criação do Ml. usando o Sequencial do Keras
classficator = Sequential()
# Adicionando a primeira camada
classficator.add(Dense(
    units=16,
    activation='relu',
    kernel_initializer='random_uniform',
    input_dim=30
))
classficator.add(Dense(
    units=16,
    activation='relu',
    kernel_initializer='random_uniform',
))
classficator.add(Dense(
    units=1,
    activation='sigmoid'
))

otimizador = keras.optimizers.Adam(
    learning_rate=0.001,
    decay=0.0001,
    clipvalue=0.5
)

#%%
# kekkei genkai


classficator.compile(
    # optimizer='adam', # eu sei, não é legal, mas preciso ver o que mudou...
    optimizer=otimizador,
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
classficator.fit(df_prev_training, df_class_training, batch_size=10, epochs=1000)

# Aplicando o conhecimento do modelo, na base de teste.

previsoes_ml2 = classficator.predict(
    df_prev_test
)
# previsoes_ml = np.array([1 if i > 0.5 else 0 for i in previsoes_ml])
previsoes_ml2 = [[1] if i > 0.5 else [0] for i in previsoes_ml2]
#previsoes_ml = (previsoes_ml > 0.5)

precisao = accuracy_score(df_class_test, previsoes_ml2)
#%%
# Matriz de confusão
matriz2 = confusion_matrix(df_class_test, previsoes_ml2)
# este é o modo manual de se avaliar, mas a biblioteca Keras tem implementações que realizam este processo.

resultado_keras2 = classficator.evaluate(df_prev_test, df_class_test)


#%%
# Considerações e Finalização


# fim
