import psycopg2
from psycopg2 import sql, Error

# Define your connection parameters
DB_PARAMS = {
    'dbname': 'Parking',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',    # e.g., 'localhost'
    'port': '5432'     # e.g., '5432'
}

try:
    # Establish the connection
    connection = psycopg2.connect(**DB_PARAMS)
    cursor = connection.cursor()

    # Print PostgreSQL Connection properties
    print("PostgreSQL server information")
    print(connection.get_dsn_parameters(), "\n")

    # Execute a query
    cursor.execute("SELECT version();")

    # Fetch the result
    db_version = cursor.fetchone()
    print("You are connected to - ", db_version, "\n")

    # Example of executing a query and fetching data
    query = sql.SQL("SELECT * FROM public.parking;")
    cursor.execute(query)
    records = cursor.fetchall()
    print(records)
    for row in records:
        print(row)

except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL", error)

finally:
    # Closing the cursor and connection
    if cursor:
        cursor.close()
    if connection:
        connection.close()
    print("PostgreSQL connection is closed")
