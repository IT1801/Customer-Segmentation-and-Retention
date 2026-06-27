import sys
from pathlib import Path
import pandas as pd

from csr.logging.logger import logging
from csr.exception.exception import CSRException
from csr.constants import (
    COL_CUSTOMER_ID,
    COL_INVOICE,
    COL_INVOICE_DATE,
    COL_PRICE,
    COL_QUANTITY,
    COL_STOCK_CODE,
    COL_DESCRIPTION,
    COL_COUNTRY,
    RAW_DATA_FILE,
)
from csr.config.configuration import ConfigurationManager, ETLConfig


def extract(
    file_path: Path = RAW_DATA_FILE,
    etl_config: ETLConfig = None,
) -> pd.DataFrame:
    """
    Load all sheets from the Online Retail II .xlsx file and
    concatenate them into a single raw DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the raw .xlsx file. Defaults to RAW_DATA_FILE
        from constants.
    etl_config : ETLConfig, optional
        ETL config object. If None, loads from ConfigurationManager.

    Returns
    -------
    pd.DataFrame
        Raw concatenated DataFrame with original column names.
        No cleaning applied.
    """
    try:
        if etl_config is None:
            etl_config = ConfigurationManager().get_etl_config()

        file_path = Path(file_path)

        # ── Validate file exists ──────────────────────────────────────────────
        if not file_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found at: {file_path}\n"
            )

        logging.info(f"Extracting raw data from: {file_path}")
        logging.info(f"Expected sheets: {etl_config.sheet_names}")

        # ── Load each sheet ───────────────────────────────────────────────────
        sheets_found = []
        failed_sheets = []
        frames = []

        for sheet in etl_config.sheet_names:
            try:
                df_sheet = pd.read_excel(
                    file_path,
                    sheet_name=sheet,
                    dtype={
                        COL_INVOICE     : str,
                        COL_STOCK_CODE  : str,
                        COL_DESCRIPTION : str,
                        COL_CUSTOMER_ID : str,
                        COL_COUNTRY     : str,
                        # Quantity and Price left as-is so pandas infers numeric
                    },
                )
                df_sheet["_source_sheet"] = sheet  # track which sheet row came from
                frames.append(df_sheet)
                sheets_found.append(sheet)
                logging.info(f"Sheet '{sheet}': {len(df_sheet):,} rows loaded")

            except Exception as e:
                # Warn and continue — don't crash if one sheet fails
                logging.warning(f"Sheet '{sheet}' could not be loaded: {e}")
                failed_sheets.append(sheet)

        # ── Validate at least one sheet loaded ────────────────────────────────
        if not frames:
            raise ValueError(
                f"No sheets could be loaded from {file_path}. "
                f"Expected: {etl_config.sheet_names}. "
                f"Failed: {failed_sheets}"
            )

        if failed_sheets:
            logging.warning(f"Sheets that failed to load: {failed_sheets}")

        # ── Concatenate all sheets ────────────────────────────────────────────
        df = pd.concat(frames, ignore_index=True)

        logging.info(f"Sheets loaded  : {sheets_found}")
        logging.info(f"Combined shape : {df.shape}")
        logging.info(f"Columns        : {df.columns.tolist()}")

        _validate_raw(df, etl_config)

        return df

    except Exception as e:
        raise CSRException(e, sys)


def _validate_raw(df: pd.DataFrame, etl_config: ETLConfig) -> None:
    """
    Lightweight sanity checks on the raw extracted DataFrame.
    Raises ValueError if critical columns are missing.
    """
    try:
        expected_columns = [
            etl_config.invoice_column,
            etl_config.stock_code_column,
            etl_config.description_column,
            etl_config.quantity_column,
            etl_config.date_column,
            etl_config.price_column,
            etl_config.customer_id_column,
            etl_config.country_column,
        ]

        missing_cols = [c for c in expected_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Raw data is missing expected columns: {missing_cols}\n"
                f"Found columns: {df.columns.tolist()}"
            )

        if df.empty:
            raise ValueError("Extracted DataFrame is empty.")

        logging.info("Raw data validation passed ✓")

    except Exception as e:
        raise CSRException(e, sys)