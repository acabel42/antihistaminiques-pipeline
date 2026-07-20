import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

database_uri = os.getenv("DB_URL")

if not database_uri:
    raise EnvironmentError("DB_URL est absente.")

engine = create_engine(database_uri)

with engine.connect() as connection:
    result = connection.execute(text("SELECT 1"))
    print(result.scalar())