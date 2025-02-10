!pip install mysql-connector-python
!pip install pymysql

import pymysql
import pandas as pd

import warnings
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


import logging

import osimport pymysql
import pandas as pd
from sklearn.model_selection import train_test_split

def installModules():
    import pymysql
    import pandas as pd

    #Exibir todas as colunas ao visualizar dataframes
    pd.set_option('display.max_columns', None)

def fetch_data_in_batches(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME, batch_size=10000):
    connection = None
    df_list = []

    try:
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USERNAME,
            password=MYSQL_PASSWORD,
            db=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("Conexão estabelecida com sucesso!")

        with connection.cursor() as cursor:
            cursor.execute(query)
            while True:
                results = cursor.fetchmany(batch_size)
                if not results:
                    break

                df_batch = pd.DataFrame(results)
                df_list.append(df_batch)

        df = pd.concat(df_list, ignore_index=True)
        return df

    except pymysql.MySQLError as e:
        print(f"Um erro ocorreu: {e}")
        return None

    finally:
        if connection:
            connection.close()
            print("Conexão fechada.")


def clean_data(df):
    df.isnull().sum() + (df == '').sum()

    #Removendo as instâncias nulas e em branco
    df = df.dropna()
    df = df[(df != '').all(axis=1)]

    #Validando a quantidade de nulos após a exclusão
    df.isnull().sum() + (df == '').sum()
    return df

def fix_data_types(df):
    df['Leucócitos'] = df['Leucócitos'].astype(float)
    df['Mielócitos'] = df['Mielócitos'].astype(float)
    df['Metamielócitos'] = df['Metamielócitos'].astype(float)
    df['Bastões'] = df['Bastões'].astype(float)
    df['Segmentados'] = df['Segmentados'].astype(float)
    df['Eosinófilos'] = df['Eosinófilos'].astype(float)
    df['Basófilos'] = df['Basófilos'].astype(float)
    df['Linfócitos'] = df['Linfócitos'].astype(float)
    df['Linfócitos Atípicos'] = df['Linfócitos Atípicos'].astype(float)
    df['Monócitos'] = df['Monócitos'].astype(float)
    df['Plasmócitos'] = df['Plasmócitos'].astype(float)
    df['Blastos'] = df['Blastos'].astype(float)
    df['Eritrócitos'] = df['Eritrócitos'].astype(float)
    df['Hemoglobina'] = df['Hemoglobina'].astype(float)
    df['Hematócrito'] = df['Hematócrito'].astype(float)
    df['HCM'] = df['HCM'].astype(float)
    df['CHCM'] = df['CHCM'].astype(float)
    df['RDW'] = df['RDW'].astype(float)
    df['Plaquetas'] = df['Plaquetas'].astype(float)
    df['MVP'] = df['MVP'].astype(float)
    df['Promielócitos'] = df['Promielócitos'].astype(float)
    df['A1C'] = df['A1C'].astype(float)
    df['Classe A1c'] = df['Classe A1c'].astype(int)
    df['Classe A1c2'] = df['Classe A1c2'].astype(int)
    return df

def remove_unusual_variables(df):
    df.drop('CodigoOs', axis=1, inplace=True)
    df.drop('Data Nascimento', axis=1, inplace=True)
    df.drop('Classe Idade', axis=1, inplace=True)
    df.drop('HCM_1', axis=1, inplace=True)
    df.drop('GME', axis=1, inplace=True)
    df.drop('G', axis=1, inplace=True)
    df.drop('Paciente', axis=1, inplace=True)
    df.drop('Data Cadastro', axis=1, inplace=True)
    df.drop('Data Cadastro Date', axis=1, inplace=True)
    df.drop('Data Nascimento Data', axis=1, inplace=True)
    return df

def remove_outliers(df):
    df = df[df['Leucócitos'] <= 200000]
    df = df[df['Mielócitos'] <= 2000]
    df = df[df['Metamielócitos'] <= 4000]
    df = df[df['Segmentados'] <= 50000]
    df = df[df['Eosinófilos'] <= 25000]
    df = df[df['Basófilos'] <= 400]
    df = df[df['Linfócitos'] <= 100000]
    df = df[df['Linfócitos Atípicos'] <= 1700]
    df = df[df['Monócitos'] <= 10000]
    df = df[df['Plasmócitos'] <= 1000]
    df = df[df['Blastos'] <= 50000]
    df = df[df['Eritrócitos'] <= 8.8]
    df = df[df['Hemoglobina'] >= 3]
    df = df[df['HCM'] <= 140]
    df = df[df['CHCM'] <= 38]
    df = df[df['MVP'] >= 3]
    df = df[df['Plaquetas'] <= 1300]
    return df

def createSubsets(df):
    X = df.drop(['Classe A1c', 'Classe A1c2', 'A1C'], axis=1)
    y = df['Classe A1c']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def get_data(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME):
    installModules()
    df = fetch_data_in_batches(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)
    df = clean_data(df)
    df = fix_data_types(df)
    df = remove_unusual_variables(df)
    df = remove_outliers(df)
    X_train, X_test, y_train, y_test = createSubsets(df)
    return X_train, X_test, y_train, y_test

def get_raw_data(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME):
    installModules()
    df = fetch_data_in_batches(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)
    df = clean_data(df)
    df = fix_data_types(df)
    df = remove_unusual_variables(df)
    df = remove_outliers(df)
    return df


#Dados de acesso à Base de dados armazenada em um MySQL

MYSQL_HOST = 'mysqlmestradogabrielmartini-mestradogabrielmartini.h.aivencloud.com'
MYSQL_PORT = 12659
MYSQL_USERNAME = 'avnadmin'
MYSQL_PASSWORD = 'AVNS_fDL87YjvKGDqrgbrww_'
DB_NAME = 'dados'

query = "SELECT * FROM dados1 limit 50000"


X_train, X_test, y_train, y_test = get_data(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)

print("Quantidade de registros", len(X_train) + len(X_test) + len(y_train) + len(y_test))


#Definir as colunas numéricas e categóricas
categorical_features = ['Sexo']
numerical_features = ['Idade', 'Leucócitos', 'Mielócitos', 'Metamielócitos', 'Bastões', 'Segmentados', 'Eosinófilos', 'Basófilos', 'Linfócitos', 'Linfócitos Atípicos', 'Monócitos', 'Plasmócitos', 'Blastos', 'Eritrócitos', 'Hemoglobina', 'Hematócrito', 'HCM', 'CHCM', 'RDW', 'Plaquetas', 'MVP', 'Promielócitos']
res_features = ['Classe A1c', 'Classe A1c2']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


def define_models(models=dict()):
  # Modelos Lineares
  models['Regressão Logística'] = LogisticRegression(max_iter=1000)
  alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  for a in alpha:
      models['Ridge Regression-'+str(a)] = RidgeClassifier(alpha=a)
  models['SGD'] = SGDClassifier(max_iter=1000, tol=1e-3)
  models['PassiveAgressive'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)

  # Modelos Não Lineares
  n_neighbors = range(1, 21)
  for k in n_neighbors:
      models['KNN-'+str(k)] = KNeighborsClassifier(n_neighbors=k)
  models['Árvore de Decisao'] = DecisionTreeClassifier()
  models['Extra Trees'] = ExtraTreeClassifier()
  models['SVM - Linear'] = SVC(kernel='linear')
  models['SVM - Polinomial'] = SVC(kernel='poly')
  models['SVM - RBF Kernel'] = SVC(kernel='rbf')
  c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  for c in c_values:
      models['SVMr-'+str(c)] = SVC(C=c)
  models['Naive Bayes'] = GaussianNB()

  # Modelos de Ensemble
  n_trees = 100
  models['AdaBoost'] = AdaBoostClassifier(n_estimators=n_trees)
  models['bagging'] = BaggingClassifier(n_estimators=n_trees)
  models['Random Forest'] = RandomForestClassifier(n_estimators=n_trees)
  models['Extra Trees'] = ExtraTreesClassifier(n_estimators=n_trees)
  models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=n_trees)

  # Redes Neurais
  models['Multilayer Perceptron'] = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,)) # Adicionando uma configuração básica de rede neural

  print('%d modelos definidos' % len(models))

  return models


# prompt: Considerando o restante do código deste notebook, crie uma função que teste vários preprocessors diferentes

import sys
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def evaluate_preprocessors(X_train, y_train, models, num_features, cat_features):
    results = []
    preprocessors = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'PowerTransformer': PowerTransformer()
    }
    for preprocessor_name, preprocessor in preprocessors.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', preprocessor, num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ])),
            ('classifier', LogisticRegression(max_iter=1000)) # Using Logistic Regression for evaluation
        ])
        cv_results = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        results.append({
            'Preprocessor': preprocessor_name,
            'Mean Accuracy': cv_results.mean(),
            'Std Accuracy': cv_results.std()
        })
    return pd.DataFrame(results)

# Assuming X_train, y_train, numerical_features, categorical_features are defined in your existing code
# Example usage (replace with your actual data and features):

models = define_models()
preprocessor_results = evaluate_preprocessors(X_train, y_train, models, numerical_features, categorical_features)
preprocessor_results


def make_pipeline(model):
    steps = list()
    steps.append(('preprocessor', preprocessor))
    steps.append(('model', model))
    pipeline = Pipeline(steps=steps)
    return pipeline

def evaluate_model(X, y, model, folds, metric):
    # Cria o pipeline conforme as etapas e modelos definidos
    pipeline = make_pipeline(model)
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
    return scores

#Validação do modelo. Caso ocorra alguma falha em dados ou hiperparâmetros, pega o erro e faz o print.
def robust_evaluate_model(X, y, model, folds, metric):
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = evaluate_model(X, y, model, folds, metric)
    except Exception as e:
        logging.error(f"Erro ao avaliar o modelo {model.__class__.__name__}: {str(e)}")
        scores = None
    return scores

#Validação do dicionário de modelos.
def evaluate_models(X, y, models, folds=10, metric='accuracy'):
    results = dict()
    for name, model in models.items():

        scores = robust_evaluate_model(X, y, model, folds, metric)

        if scores is not None:

            results[name] = scores
            mean_score, std_score = mean(scores), std(scores)
            print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
        else:
            print('>%s: erro' % name)
    return results

def summarize_results(results, maximize=True, top_n=10):

    if len(results) == 0:
        print('Sem Resultados')
        return

    n = min(top_n, len(results))

    mean_scores = [(k,mean(v)) for k,v in results.items()]

    mean_scores = sorted(mean_scores, key=lambda x: x[1])

    if maximize:
        mean_scores = list(reversed(mean_scores))

    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]

    print()
    for i in range(n):
        name = names[i]
        mean_score, std_score = mean(results[name]), std(results[name])
        print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))

    pyplot.boxplot(scores, labels=names)
    _, labels = pyplot.xticks()
    pyplot.setp(labels, rotation=90)
    pyplot.savefig('spotcheck.png')


#Resultados com todas instaâncias

models = define_models()

results = evaluate_models(X_train, y_train, models)

summarize_results(results)