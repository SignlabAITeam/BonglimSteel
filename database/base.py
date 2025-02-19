import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

load_dotenv(r"C:\Users\signlab_039\Desktop\projects\bonglim\backend\database\.env")

USER = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')
NAME = os.getenv('DB_NAME')
HOST = os.getenv('DB_HOST')
PORT = os.getenv('DB_PORT')

SQLALCHEMY_DATABASE_URL = f'mysql+mysqldb://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}'

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()