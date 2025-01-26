def my_function():
    print("Hello from my_function in my_module!")
    return 1

def installModules():
    import pymysql
    import pandas as pd

    #Exibir todas as colunas ao visualizar dataframes
    pd.set_option('display.max_columns', None)