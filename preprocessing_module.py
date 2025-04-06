import pymysql
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

def createTrainSubset(df):
    X = df.drop(['Classe A1c', 'Classe A1c2', 'A1C'], axis=1)
    y = df['Classe A1c2']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, y_train

def createTestSubset(df):
    X = df.drop(['Classe A1c', 'Classe A1c2', 'A1C'], axis=1)
    y = df['Classe A1c2']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_test, y_test

def get_train_data(limit, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME):
    installModules()
    query = f"select * from dados1 where split = 'train' ", limit
    df = fetch_data_in_batches(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)
    df = clean_data(df)
    df = fix_data_types(df)
    df = remove_unusual_variables(df)
    df = remove_outliers(df)
    X_train, y_train = createTrainSubset(df)

    return  X_train, y_train

def get_test_data(limit, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME):
    installModules()
    query = f"select * from dados1 where split = 'test'", limit
    df = fetch_data_in_batches(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)
    df = clean_data(df)
    df = fix_data_types(df)
    df = remove_unusual_variables(df)
    df = remove_outliers(df)

    X_test, y_test = createTestSubset(df)
    return X_test, y_test



def get_raw_data(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME):
    installModules()
    df = fetch_data_in_batches(query, MYSQL_HOST, MYSQL_PORT, MYSQL_USERNAME, MYSQL_PASSWORD, DB_NAME)
    df = clean_data(df)
    df = fix_data_types(df)
    df = remove_unusual_variables(df)
    df = remove_outliers(df)
    return df