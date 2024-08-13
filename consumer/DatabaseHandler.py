import psycopg2
from psycopg2 import sql
from Configuration import Configuration as CONFIG
from psycopg2.extras import Json

class DatabaseHandler:
    def __init__(self):
        self._db_params = {
                'dbname': CONFIG.DB_NAME,
                'user': CONFIG.DB_USER,
                'password': CONFIG.DB_PASSWORD,
                'host': CONFIG.DB_HOST,
                'port': CONFIG.DB_PORT,    
                }
        
    def get_last_not_analyzed_index(self, timestamp):
        query = f"""SELECT * FROM {CONFIG.DB_TABLE_NAME} 
                    WHERE is_analyzed=false
                    AND timestamp='{timestamp}' 
                    ORDER BY index DESC;
                """
        connection = psycopg2.connect(**self._db_params)
        cursor = connection.cursor()
        query = sql.SQL(query)
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        connection.close()

        latest_index = [r[0] for r in records][0]
        
        return latest_index

    def update_frame_stats(self, index, timestamp, stat):
        connection = psycopg2.connect(**self._db_params)
        cursor = connection.cursor()

        query = sql.SQL(f"""
            UPDATE {CONFIG.DB_TABLE_NAME}
            SET stat = %s, is_analyzed=true
            WHERE index = %s
            AND timestamp = %s;
        """)

        cursor.execute(query, (Json(stat), index, timestamp))
        connection.commit()

        cursor.close()
        connection.close()
        
    
    def push_frame_to_db(self, index, timestamp):
        connection = psycopg2.connect(**self._db_params)
        cursor = connection.cursor()

        query = sql.SQL(f"""
            INSERT INTO {CONFIG.DB_TABLE_NAME} (index, timestamp, is_analyzed)
            VALUES (%s, %s, %s);
        """)

        cursor.execute(query, (index, timestamp, False))
        connection.commit()

        cursor.close()
        connection.close()
    
    def is_there_the_frame(self, index, timestamp):
        query = f"""SELECT * FROM {CONFIG.DB_TABLE_NAME} 
                    WHERE index='{index}'
                    AND timestamp='{timestamp}'
                    AND is_analyzed=true
                """
        connection = psycopg2.connect(**self._db_params)
        cursor = connection.cursor()
        query = sql.SQL(query)
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        connection.close()

        if len(records) == 1:
            return True
        elif len(records) == 0:
            return False
        else:
            raise RuntimeError("WTF!")