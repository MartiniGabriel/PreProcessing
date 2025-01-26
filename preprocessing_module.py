import pymysql

def my_function():
    print("Hello from my_function in my_module!")
    return 1

def installModules():
    import pymysql
    import pandas as pd

    #Exibir todas as colunas ao visualizar dataframes
    pd.set_option('display.max_columns', None)

def fetch_data_in_batches(query, host, port, username, password, db, batch_size=10000):
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
