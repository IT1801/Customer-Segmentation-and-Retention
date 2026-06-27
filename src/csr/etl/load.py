"""
Responsible for loading the cleaned DataFrame into PostgreSQL.
Creates the schema if it doesn't exist, then writes the cleaned
transactions table using SQLAlchemy + psycopg2.

Nothing is cleaned or transformed here — load does one thing:
persist the DataFrame to the database.
"""

import sys

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from logging.logger import logging
from exception.exception import CSRException
from src.csr.constants import (
    DB_SCHEMA,
    TABLE_CLEANED_TRANSACTIONS,
    DB_INSERT_CHUNKSIZE,
)
from src.csr.config.configuration import ConfigurationManager, DatabaseConfig


# ─── Engine factory ───────────────────────────────────────────────────────────

def get_engine(db_config: DatabaseConfig = None) -> Engine:
    """
    Build and return a SQLAlchemy engine from DatabaseConfig.

    Parameters
    ----------
    db_config : DatabaseConfig, optional
        If None, loads from ConfigurationManager.

    Returns
    -------
    Engine
        SQLAlchemy engine connected to PostgreSQL.
    """
    try:
        if db_config is None:
            db_config = ConfigurationManager().get_database_config()

        engine = create_engine(
            db_config.url,
            pool_pre_ping=True,    # test connection before using from pool
            pool_size=5,
            max_overflow=10,
        )

        # Verify connection is reachable
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logging.info(f"Database connection established → {db_config.host}:{db_config.port}/{db_config.name}")
        return engine

    except Exception as e:
        raise CSRException(e, sys)


# ─── Schema setup ─────────────────────────────────────────────────────────────

def _create_schema(engine: Engine, schema: str = DB_SCHEMA) -> None:
    """Create the database schema if it does not already exist."""
    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        logging.info(f"Schema '{schema}' ready")
    except Exception as e:
        raise CSRException(e, sys)


# ─── Main load function ───────────────────────────────────────────────────────

def save_interim(
    df: pd.DataFrame,
    engine: Engine = None,
    schema: str = DB_SCHEMA,
    table: str = TABLE_CLEANED_TRANSACTIONS,
    chunksize: int = DB_INSERT_CHUNKSIZE,
    if_exists: str = "replace",
) -> None:
    """
    Write the cleaned transactions DataFrame to PostgreSQL.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame returned by transform().
    engine : Engine, optional
        SQLAlchemy engine. If None, builds one from ConfigurationManager.
    schema : str
        Postgres schema name. Defaults to DB_SCHEMA constant.
    table : str
        Target table name. Defaults to TABLE_CLEANED_TRANSACTIONS constant.
    chunksize : int
        Number of rows per INSERT batch. Defaults to DB_INSERT_CHUNKSIZE.
    if_exists : str
        Pandas behaviour if table exists — 'replace' or 'append'.
        Use 'replace' for full refresh, 'append' for incremental loads.
    """
    try:
        if engine is None:
            engine = get_engine()

        _create_schema(engine, schema)

        full_table = f"{schema}.{table}"
        logging.info(
            f"Loading {len(df):,} rows → {full_table} "
            f"(if_exists='{if_exists}', chunksize={chunksize:,})"
        )

        df.to_sql(
            name      = table,
            con       = engine,
            schema    = schema,
            if_exists = if_exists,
            index     = False,
            chunksize = chunksize,
            method    = "multi",   # batch rows into single INSERT statements
        )

        _verify_load(engine, schema, table, expected_rows=len(df))

    except Exception as e:
        raise CSRException(e, sys)


# ─── Post-load verification ───────────────────────────────────────────────────

def _verify_load(
    engine: Engine,
    schema: str,
    table: str,
    expected_rows: int,
) -> None:
    """
    Query the table after loading and confirm row count matches.
    Raises if row count does not match expected.
    """
    try:
        full_table = f"{schema}.{table}"

        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM {full_table}")
            )
            actual_rows = result.scalar()

        if actual_rows != expected_rows:
            raise ValueError(
                f"Row count mismatch in {full_table}: "
                f"expected {expected_rows:,}, found {actual_rows:,}"
            )

        logging.info(
            f"Load verified ✓ — {actual_rows:,} rows in {full_table}"
        )

    except Exception as e:
        raise CSRException(e, sys)