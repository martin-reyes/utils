import pandas as pd
import env
import os


def get_connection(db, user=env.user, host=env.host, password=env.pwd):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    

def get_telco_data(sql_query, db, filename):
    '''
    If the csv file exists, it is read and returned as a pandas DataFrame
    If not, pandas reads in a SQL query that acquires telco customer data from a MySQL database.
    The query is stored into a DataFrame, saved, and returned.
    '''
    # Read the csv file if it exists
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # Fead the SQL query into a dataframe
        df = pd.read_sql(sql_query,
                         get_connection(db))
        # Write that DataFrame for prep
        df.to_csv(filename, index=False)
        # Return the DataFrame
        return df