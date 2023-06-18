import env
import os

def get_connection(db, user=env.user, host=env.host, password=env.pwd):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'