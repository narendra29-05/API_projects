import pandas as pd
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

# Function to create SQLite database from CSV files
def create_db_from_csvs(uploaded_files, data_dir):
    # Create a unique database name
    db_name = f"user_data_{uuid.uuid4().hex[:8]}.db"
    db_path = data_dir / db_name
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    
    # Dictionary to store table schemas
    schemas = {}
    # Dictionary to store dataframes
    dataframes = {}
    # Lists to track file names
    csv_files = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Get file name
        file_name = uploaded_file.name
        csv_files.append(file_name)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            # Write the uploaded file content to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Clean table name (remove extension and special characters)
        table_name = os.path.splitext(file_name)[0]
        table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
        
        # Store dataframe for preview
        dataframes[table_name] = df
        
        # Write DataFrame to SQLite
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        
        # Generate schema information for this table
        column_info = []
        for column in df.columns:
            dtype = str(df[column].dtype)
            if "int" in dtype:
                col_type = "INTEGER"
            elif "float" in dtype:
                col_type = "REAL"
            elif "datetime" in dtype:
                col_type = "TIMESTAMP"
            else:
                col_type = "TEXT"
            column_info.append(f"{column} ({col_type})")
        
        # Store schema
        schemas[table_name] = column_info
        
        # Clean up the temporary file
        os.unlink(file_path)
    
    conn.close()
    
    # Format schema string for the LLM
    schema_str = ""
    for table, columns in schemas.items():
        schema_str += f"Table: {table}\n"
        schema_str += f"Columns: {', '.join(columns)}\n\n"
    
    return str(db_path), schema_str, dataframes

# Function to execute SQL and return results
def execute_sql(db_path, sql_query):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return str(e)