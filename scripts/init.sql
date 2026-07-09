-- scripts/init.sql
-- Runs automatically when the Postgres container starts for the first time.
-- Creates the retail schema so the ETL pipeline can write tables immediately.

CREATE SCHEMA IF NOT EXISTS retail;

COMMENT ON SCHEMA retail IS
  'Customer segmentation and retention analysis — Online Retail II dataset';