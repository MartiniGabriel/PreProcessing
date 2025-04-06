import os
import sys


import pymysql
import pandas as pd
from sklearn.model_selection import train_test_split

# Redireciona os prints para um arquivo de texto
log_file = "output.txt"
sys.stdout = open(log_file, "w")

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

def createTrainSubset(df):
    X = df.drop(['Classe A1c', 'Classe A1c2', 'A1C'], axis=1)
    y = df['Classe A1c2']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, y_train

def createTrainTestSubsets(df):
    X = df.drop(['Classe A1c', 'Classe A1c2', 'A1C'], axis=1)
    y = df['Classe A1c2']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def get_train_data(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME):
    installModules()
    df = fetch_data_in_batches(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)
    df = clean_data(df)
    df = fix_data_types(df)
    df = remove_unusual_variables(df)
    df = remove_outliers(df)

    X_train, y_train = createTrainSubset(df)
    return  X_train, y_train

def get_train_test_data(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME):
    installModules()
    df = fetch_data_in_batches(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)
    df = clean_data(df)
    df = fix_data_types(df)
    df = remove_unusual_variables(df)
    df = remove_outliers(df)

    X_train, X_test, y_train, y_test = createTrainTestSubsets(df)
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

query = "SELECT * FROM dados1 limit 1000"

X_train, X_test, y_train, y_test = get_train_test_data(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)

print("Quantidade de registros", len(X_train) + len(X_test) + len(y_train) + len(y_test))


from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


categorical_features = ['Sexo']
numerical_features = ['Idade', 'Leucócitos', 'Mielócitos', 'Metamielócitos', 'Bastões', 'Segmentados', 'Eosinófilos', 'Basófilos', 'Linfócitos', 'Linfócitos Atípicos', 'Monócitos', 'Plasmócitos', 'Blastos', 'Eritrócitos', 'Hemoglobina', 'Hematócrito', 'HCM', 'CHCM', 'RDW', 'Plaquetas', 'MVP', 'Promielócitos']
res_features = ['Classe A1c', 'Classe A1c2']
set_config(display="diagram")

preprocessor_StandardScaler_OneHotEncoder = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

preprocessor_MinMaxScaler_OneHotEncoder = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder = ColumnTransformer(
    transformers=[
        ('num_standard', StandardScaler(), numerical_features),
        ('num_minmax', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

from sklearn.pipeline import Pipeline

def make_pipeline(model, preprocessor):
    steps = list()
    steps.append(('preprocessor', preprocessor))
    steps.append(('classifier', model))
    pipeline = Pipeline(steps=steps)
    return pipeline


import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV


def run_ncv(pipeline, grid_params, X_train, y_train):

    scoring = {
        'recall': 'recall',
        'precision': 'precision',
        'accuracy': 'accuracy',
        'F2': make_scorer(fbeta_score, beta=2),
        'AUC-PR': 'average_precision',
        'ROC-AUC': 'roc_auc'
    }

    ### loop interno ####
    cv_inner = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    f2_scorer = make_scorer(fbeta_score, beta=2)

    search = HalvingRandomSearchCV(
        estimator=pipeline,
        param_distributions=grid_params,
        scoring=f2_scorer,
        n_jobs=-1,
        cv=cv_inner,
        refit=f2_scorer,
        verbose=2,
        factor=3,
        random_state=42
    )
    ### loop interno ####

    ### loop externo ####
    cv_outer = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    # executa o nested cross-validation
    output_ncv = cross_validate(search, X_train, y_train, scoring=scoring, cv=cv_outer, n_jobs=-1,return_estimator=True, return_train_score=True)


    return output_ncv, scoring


from statistics import mode
import numpy as np

def print_results(pipeline, output_NCV):
    model_name = pipeline.named_steps['classifier'].__class__.__name__
    print("Model name:", model_name)

    for metric in scoring.keys():
        train_key = f"train_{metric}"
        test_key = f"test_{metric}"

        if train_key in results:
            train_mean = np.mean(results[train_key])
            train_std = np.std(results[train_key])
            print(f"Train {metric}: Mean = {train_mean:.2f}, Std = {train_std:.2f}")
        if test_key in results:
            test_mean = np.mean(results[test_key])
            test_std = np.std(results[test_key])
            print(f"Test {metric}: Mean = {test_mean:.2f}, Std = {test_std:.2f}")
        print()

    values = [{key: value for key, value in x.best_params_.items()} for x in output_NCV['estimator']]
    modes = {key: mode(d[key] for d in values) for key in values[0]}

    best_params = modes
    print("\nBest Params:", best_params)

    # Retorna um dicionário com melhores parâmetros para treinamento do modelo final
    best_params_dic = {key.replace('classifier__', '', 1): value for key, value in best_params.items()}
    return best_params_dic



from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score

def best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test):
    pipeline_best_model.fit(X_train, y_train)

    y_pred = pipeline_best_model.predict(X_test)

    if hasattr(pipeline_best_model, "predict_proba"):
        y_prob = pipeline_best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None



    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    if y_prob is not None:
        auc_pr = average_precision_score(y_test, y_prob)
    else:
        auc_pr = None


    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Exibe as métricas
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F2 Score: {f2:.2f}")
    if auc_pr is not None:
        print(f"AUC-PR: {auc_pr:.2f}")
    print("\nConfusion Matrix:")
    print(confusion)
    print("\nClassification Report:")
    print(classification_rep)

    # Retorna um dicionário com as métricas calculadas
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "auc_pr": auc_pr,
        "confusion_matrix": confusion,
        "classification_report": classification_rep
    }

    return results


def get_bagging_parameters(params):
    params_estimator = {}
    params_bagging = {}
    for chave, valor in params.items():
        if chave.startswith('estimator__'):
            params_estimator[chave] = valor
        else:
            params_bagging[chave] = valor
    params_bagging.pop('estimator', None)
    params_estimator = {chave.removeprefix('estimator__'): valor for chave, valor in params_estimator.items()}
    return params_estimator, params_bagging


# Define o grid de parâmetros para otimização
grid_params = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 17, 23, 29, 31, 33, 37, 41],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
    }

print("KNN + StandardScaler + OneHotEncoder")

from sklearn.neighbors import KNeighborsClassifier

pipeline = make_pipeline(KNeighborsClassifier(), preprocessor_StandardScaler_OneHotEncoder)

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(KNeighborsClassifier(**best_params), preprocessor_StandardScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)



print("KNN + MinMaxScaler + OneHotEncoder")
# Define o grid de parâmetros para otimização
grid_params = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 17, 23, 29, 31, 33, 37, 41],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
    }

from sklearn.neighbors import KNeighborsClassifier

pipeline = make_pipeline(KNeighborsClassifier(), preprocessor_MinMaxScaler_OneHotEncoder)

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(KNeighborsClassifier(**best_params), preprocessor_MinMaxScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)


print("KNN + StandardScaler + MinMaxScaler + OneHotEncoder")
# Define o grid de parâmetros para otimização
grid_params = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 17, 23, 29, 31, 33, 37, 41],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
    }

from sklearn.neighbors import KNeighborsClassifier

pipeline = make_pipeline(KNeighborsClassifier(), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(KNeighborsClassifier(**best_params), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)


print("Multilayer Perceptron + StandardScaler + MinMax Scaler + OneHotEncoder")

grid_params = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'classifier__activation': ['relu', 'tanh', 'logistic'],
    'classifier__solver': ['adam', 'sgd'],
    # Taxa de regularização L2:
    'classifier__alpha': [1e-4, 1e-3, 1e-2],
    'classifier__learning_rate': ['constant', 'adaptive'],
    'classifier__learning_rate_init': [0.001, 0.01, 0.1],
    'classifier__max_iter': [200, 400, 600]
    }

from sklearn.neural_network import MLPClassifier

pipeline = make_pipeline(MLPClassifier(), preprocessor_StandardScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(MLPClassifier(**best_params), preprocessor_StandardScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)


print("Multilayer Perceptron + StandardScaler + OneHotEncoder")

grid_params = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'classifier__activation': ['relu', 'tanh', 'logistic'],
    'classifier__solver': ['adam', 'sgd'],
    # Taxa de regularização L2:
    'classifier__alpha': [1e-4, 1e-3, 1e-2],
    'classifier__learning_rate': ['constant', 'adaptive'],
    'classifier__learning_rate_init': [0.001, 0.01, 0.1],
    'classifier__max_iter': [200, 400, 600]
    }

from sklearn.neural_network import MLPClassifier

pipeline = make_pipeline(MLPClassifier(), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(MLPClassifier(**best_params), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)

print("Árvore de Decisão + StandardScaler + OneHotEncoder")

grid_params = {
    'classifier__criterion': ['gini', 'entropy'],  # Funções de medida de impureza
    'classifier__splitter': ['best', 'random'],    # Estratégias de divisão
    'classifier__max_depth': [None, 10, 20, 30],   # Profundidade máxima da árvore
    'classifier__min_samples_split': [2, 10, 20],  # Número mínimo de amostras para dividir um nó
    'classifier__min_samples_leaf': [1, 5, 10],    # Número mínimo de amostras em uma folha
    'classifier__max_features': [None, 'sqrt', 'log2'],  # Número de features a considerar ao procurar a melhor divisão
    'classifier__ccp_alpha': [0.0, 0.01, 0.1]      # Parâmetro de complexidade para poda mínima de custo-complexidade
    }

from sklearn.tree import DecisionTreeClassifier

pipeline = make_pipeline(DecisionTreeClassifier(), preprocessor_StandardScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(DecisionTreeClassifier(**best_params), preprocessor_StandardScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)


print("Árvore de Decisão + StandardScaler + MinMaxScaler + OneHotEncoder")
grid_params = {
    'classifier__criterion': ['gini', 'entropy'],  # Funções de medida de impureza
    'classifier__splitter': ['best', 'random'],    # Estratégias de divisão
    'classifier__max_depth': [None, 10, 20, 30],   # Profundidade máxima da árvore
    'classifier__min_samples_split': [2, 10, 20],  # Número mínimo de amostras para dividir um nó
    'classifier__min_samples_leaf': [1, 5, 10],    # Número mínimo de amostras em uma folha
    'classifier__max_features': [None, 'sqrt', 'log2'],  # Número de features a considerar ao procurar a melhor divisão
    'classifier__ccp_alpha': [0.0, 0.01, 0.1]      # Parâmetro de complexidade para poda mínima de custo-complexidade
    }

from sklearn.tree import DecisionTreeClassifier

pipeline = make_pipeline(DecisionTreeClassifier(), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(DecisionTreeClassifier(**best_params), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)

print("Árvore de Decisão + MinMaxScaler + OneHotEncoder")

grid_params = {
    'classifier__criterion': ['gini', 'entropy'],  # Funções de medida de impureza
    'classifier__splitter': ['best', 'random'],    # Estratégias de divisão
    'classifier__max_depth': [None, 10, 20, 30],   # Profundidade máxima da árvore
    'classifier__min_samples_split': [2, 10, 20],  # Número mínimo de amostras para dividir um nó
    'classifier__min_samples_leaf': [1, 5, 10],    # Número mínimo de amostras em uma folha
    'classifier__max_features': [None, 'sqrt', 'log2'],  # Número de features a considerar ao procurar a melhor divisão
    'classifier__ccp_alpha': [0.0, 0.01, 0.1]      # Parâmetro de complexidade para poda mínima de custo-complexidade
    }

from sklearn.tree import DecisionTreeClassifier

pipeline = make_pipeline(DecisionTreeClassifier(), preprocessor_MinMaxScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(DecisionTreeClassifier(**best_params), preprocessor_MinMaxScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)


print("Bagging + MinMaxScaler + OneHotEncoder")

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

grid_params = {
    'classifier__n_estimators': [10, 50, 100],           # Número de estimadores no ensemble
    'classifier__max_samples': [0.5, 0.7, 1.0],          # Proporção de amostras para treinar cada estimador base
    'classifier__max_features': [0.5, 0.7, 1.0],         # Proporção de features para treinar cada estimador base
    'classifier__bootstrap': [True, False],              # Se as amostras são extraídas com substituição
    'classifier__bootstrap_features': [True, False],     # Se as features são extraídas com substituição
    'classifier__estimator': [DecisionTreeClassifier()],   # Estimador base
    'classifier__estimator__criterion': ['gini', 'entropy'],  # Critério de divisão do DecisionTreeClassifier
    'classifier__estimator__max_depth': [None, 10, 20, 30]    # Profundidade máxima do DecisionTreeClassifier
    }

pipeline = make_pipeline(BaggingClassifier(), preprocessor_MinMaxScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

estimator_params, bagging_params = get_bagging_parameters(best_params)

from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(**estimator_params)

pipeline_best_model = make_pipeline(BaggingClassifier(estimator=base_estimator, **bagging_params), preprocessor_MinMaxScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)


print("Bagging + StandardScaler + OneHotEncoder")

grid_params = {
    'classifier__n_estimators': [10, 50, 100],           # Número de estimadores no ensemble
    'classifier__max_samples': [0.5, 0.7, 1.0],          # Proporção de amostras para treinar cada estimador base
    'classifier__max_features': [0.5, 0.7, 1.0],         # Proporção de features para treinar cada estimador base
    'classifier__bootstrap': [True, False],              # Se as amostras são extraídas com substituição
    'classifier__bootstrap_features': [True, False],     # Se as features são extraídas com substituição
    'classifier__estimator': [DecisionTreeClassifier()],   # Estimador base
    'classifier__estimator__criterion': ['gini', 'entropy'],  # Critério de divisão do DecisionTreeClassifier
    'classifier__estimator__max_depth': [None, 10, 20, 30]    # Profundidade máxima do DecisionTreeClassifier
    }

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


pipeline = make_pipeline(BaggingClassifier(), preprocessor_StandardScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

estimator_params, bagging_params = get_bagging_parameters(best_params)

from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(**estimator_params)


pipeline_best_model = make_pipeline(BaggingClassifier(estimator=base_estimator, **bagging_params), preprocessor_StandardScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)


print("Bagging + StandardSacler + MinMaxScaler + OneHotEncoder")

grid_params = {
    'classifier__n_estimators': [10, 50, 100],           # Número de estimadores no ensemble
    'classifier__max_samples': [0.5, 0.7, 1.0],          # Proporção de amostras para treinar cada estimador base
    'classifier__max_features': [0.5, 0.7, 1.0],         # Proporção de features para treinar cada estimador base
    'classifier__bootstrap': [True, False],              # Se as amostras são extraídas com substituição
    'classifier__bootstrap_features': [True, False],     # Se as features são extraídas com substituição
    'classifier__estimator': [DecisionTreeClassifier()],   # Estimador base
    'classifier__estimator__criterion': ['gini', 'entropy'],  # Critério de divisão do DecisionTreeClassifier
    'classifier__estimator__max_depth': [None, 10, 20, 30]    # Profundidade máxima do DecisionTreeClassifier
    }

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


pipeline = make_pipeline(BaggingClassifier(), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

estimator_params, bagging_params = get_bagging_parameters(best_params)

from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(**estimator_params)


pipeline_best_model = make_pipeline(BaggingClassifier(estimator=base_estimator, **bagging_params), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)



print("Gradient Boosting + MinaMaxScaler + OneHotEncoder")
grid_params = {
    'classifier__loss': ['exponential', 'log_loss'],                 # Função de perda: 'deviance' para log-loss, 'exponential' para AdaBoost
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],       # Taxa de aprendizado
    'classifier__n_estimators': [50, 100, 200],                     # Número de estimadores (árvores)
    'classifier__subsample': [0.8, 0.9, 1.0],                       # Fração de amostras utilizadas para treinar cada árvore
    'classifier__criterion': ['squared_error', 'friedman_mse'],              # Critério de divisão: 'friedman_mse' ou 'mae'
    'classifier__max_depth': [3, 4, 5, 6],                           # Profundidade máxima das árvores
    'classifier__min_samples_split': [2, 5, 10],                    # Número mínimo de amostras para dividir um nó
    'classifier__min_samples_leaf': [1, 2, 4],                      # Número mínimo de amostras em um nó folha
    'classifier__max_features': ['sqrt', 'log2', None]            # Número de features a serem consideradas para a melhor divisão
}

from sklearn.ensemble import GradientBoostingClassifier


pipeline = make_pipeline(GradientBoostingClassifier(warm_start=True), preprocessor_MinMaxScaler_OneHotEncoder)

pipeline

results, scoring = run_ncv(pipeline, grid_params, X_train, y_train)
pd.DataFrame(results)

best_params = print_results(pipeline, results)

pipeline_best_model = make_pipeline(GradientBoostingClassifier(**best_params), preprocessor_StandardScaler_MinMaxScaler_OneHotEncoder)

pipeline_best_model

results_json = best_model_fit_predict(pipeline_best_model, X_train, y_train, X_test, y_test)


# Fecha o redirecionamento do arquivo ao final do script
sys.stdout.close()
sys.stdout = sys.__stdout__