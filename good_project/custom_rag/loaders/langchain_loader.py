from langchain.schema import Document
import os
import sqlite3
import pandas as pd
from snowflake import connector
import psycopg2

class ParentChildLangchainLoader:
    """
    A loader class responsible for reading data from various sources
    and converting it into a list of LangChain Document objects.
    This version handles text files, SQLite, Snowflake, and PostgreSQL.
    """
    def __init__(self, *args, **kwargs):
        pass

    def load_from_file(self, file_path: str) -> list[Document]:
        """Loads data from a single local file (text, csv, or SQLite)."""
        docs = []
        file_extension = os.path.splitext(file_path)[1]
        file_name = os.path.basename(file_path)

        if file_extension in [".db", ".sqlite", ".sqlite3"]:
            try:
                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for table_name_tuple in tables:
                    table_name = table_name_tuple[0]
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    for index, row in df.iterrows():
                        content = ", ".join([f"{col}: {val}" for col, val in row.astype(str).items()])
                        docs.append(Document(page_content=content, metadata={"source": file_name, "table": table_name, "row": index}))
                conn.close()
            except Exception as e:
                print(f"Error processing database file {file_name}: {e}")
        else:
            try:
                # This handles .txt, .md, .csv, etc.
                df = pd.read_csv(file_path)
                for index, row in df.iterrows():
                    content = ", ".join([f"{col}: {val}" for col, val in row.astype(str).items()])
                    docs.append(Document(page_content=content, metadata={"source": file_name, "row": index}))
            except Exception:
                # Fallback for non-csv text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                docs.append(Document(page_content=content, metadata={"source": file_name}))
        
        return docs

    def load_from_snowflake(self, user, password, account, database, schema, table) -> list[Document]:
        """Loads data from a Snowflake table."""
        docs = []
        try:
            with connector.connect(
                user=user, password=password, account=account,
                database=database, schema=schema
            ) as conn:
                query = f"SELECT * FROM {database}.{schema}.{table}"
                df = pd.read_sql(query, conn)
            
            for index, row in df.iterrows():
                content = ", ".join([f"{col}: {val}" for col, val in row.astype(str).items()])
                docs.append(Document(page_content=content, metadata={"source": "Snowflake", "table": table, "row": index}))
        except Exception as e:
            print(f"Error connecting to or fetching from Snowflake: {e}")
            raise e
            
        return docs

    def load_from_postgres(self, user, password, host, port, dbname, table) -> list[Document]:
        """Loads data from a PostgreSQL table."""
        docs = []
        try:
            conn_string = f"dbname='{dbname}' user='{user}' host='{host}' port='{port}' password='{password}'"
            with psycopg2.connect(conn_string) as conn:
                query = f'SELECT * FROM "{table}"' # Use quotes for table name
                df = pd.read_sql_query(query, conn)

            for index, row in df.iterrows():
                content = ", ".join([f"{col}: {val}" for col, val in row.astype(str).items()])
                docs.append(Document(page_content=content, metadata={"source": "PostgreSQL", "table": table, "row": index}))
        except Exception as e:
            print(f"Error connecting to or fetching from PostgreSQL: {e}")
            raise e
            
        return docs

    def load_and_convert(self, folder_path, shortlisted_files):
        """Original method from your trainer script, adapted for file loading."""
        all_docs = []
        for file_path in shortlisted_files:
            all_docs.extend(self.load_from_file(file_path))
        return all_docs, shortlisted_files, [], {}, {}
