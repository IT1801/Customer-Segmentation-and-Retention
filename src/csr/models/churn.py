"""
XGBoost churn prediction model with SHAP explainability.

Steps:
    1. Load segment results from PostgreSQL
    2. Define churn label (Recency > CHURN_THRESHOLD_DAYS)
    3. Train/test split (stratified)
    4. Fit XGBoost classifier
    5. Evaluate — ROC-AUC, classification report
    6. Compute SHAP values
    7. Log everything to MLflow
    8. Save model artifact
    9. Write churn predictions back to PostgreSQL

Run directly:
    python -m src.csr.models.churn
"""

import sys
import time

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sqlalchemy import text

from csr.exception.exception import CSRException
from csr.logging.logger import logging
from src.csr.config.configuration import ConfigurationManager
from src.csr.constants import (
    ARTIFACTS_DIR,
    CHURN_COLSAMPLE_BYTREE,
    CHURN_FEATURE_COLUMNS,
    CHURN_LEARNING_RATE,
    CHURN_MAX_DEPTH,
    CHURN_MODEL_ARTIFACT,
    CHURN_N_ESTIMATORS,
    CHURN_RISK_BINS,
    CHURN_RISK_LABELS,
    CHURN_SUBSAMPLE,
    CHURN_TEST_SIZE,
    CHURN_THRESHOLD_DAYS,
    COL_CUSTOMER_ID,
    DB_INSERT_CHUNKSIZE,
    DB_SCHEMA,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    TABLE_CHURN_PREDICTIONS,
    TABLE_SEGMENT_RESULTS,
)
from src.csr.etl.load import get_engine


def run_churn_pipeline() -> None:
    """
    Run the full churn prediction pipeline end to end.
    """
    try:
        start = time.time()
        logging.info("=" * 60)
        logging.info("CHURN PIPELINE STARTED")
        logging.info("=" * 60)

        # ── Step 0: Config & engine ───────────────────────────────────────────
        cfg        = ConfigurationManager()
        db_cfg     = cfg.get_database_config()
        churn_cfg  = cfg.get_churn_config()
        engine     = get_engine(db_config=db_cfg)

        # ── MLflow setup ──────────────────────────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="xgboost_churn"):

            # ── Step 1: Load data ─────────────────────────────────────────────
            logging.info("Step 1/6 — Loading segment results from Postgres")
            df = _load_segment_results(engine)

            # ── Step 2: Prepare features & label ─────────────────────────────
            logging.info(
                f"Step 2/6 — Defining churn label "
                f"(Recency > {CHURN_THRESHOLD_DAYS} days)"
            )
            X, y = _prepare_data(df, churn_cfg.feature_columns)

            # ── Step 3: Train/test split ──────────────────────────────────────
            logging.info(
                f"Step 3/6 — Train/test split "
                f"(test_size={churn_cfg.test_size})"
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size    = churn_cfg.test_size,
                random_state = churn_cfg.random_state,
                stratify     = y,
            )
            logging.info(
                f"Train: {X_train.shape} | "
                f"Test: {X_test.shape} | "
                f"Churn rate: {y.mean()*100:.1f}%"
            )

            # ── Step 4: Fit XGBoost ───────────────────────────────────────────
            logging.info("Step 4/6 — Fitting XGBoost")
            model = _fit_xgboost(
                X_train, y_train,
                X_test, y_test,
                churn_cfg,
            )

            # ── Step 5: Evaluate ──────────────────────────────────────────────
            logging.info("Step 5/6 — Evaluating model")
            metrics = _evaluate(model, X_test, y_test)

            # ── Step 6: SHAP + MLflow + artifacts + save ──────────────────────
            logging.info("Step 6/6 — SHAP, MLflow, artifacts, saving predictions")
            shap_values = _compute_shap(model, X_test)
            _log_mlflow(churn_cfg, metrics, shap_values, X_test)
            _save_artifact(model)
            _save_predictions(df, model, X, engine)

        elapsed = time.time() - start
        logging.info("=" * 60)
        logging.info("CHURN PIPELINE COMPLETE")
        logging.info(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
        logging.info(f"  PR-AUC       : {metrics['pr_auc']:.4f}")
        logging.info(f"  Total time   : {elapsed:.1f}s")
        logging.info("=" * 60)

    except Exception as e:
        logging.error("CHURN PIPELINE FAILED")
        raise CSRException(e, sys)


# ─── Step 1: Load segment results ─────────────────────────────────────────────

def _load_segment_results(engine) -> pd.DataFrame:
    try:
        query = text(
            f"SELECT * FROM {DB_SCHEMA}.{TABLE_SEGMENT_RESULTS}"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        logging.info(
            f"Segment results loaded — "
            f"{len(df):,} customers × {df.shape[1]} columns"
        )
        return df

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 2: Prepare features & churn label ───────────────────────────────────

def _prepare_data(
    df: pd.DataFrame,
    feature_columns: list,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Define churn label and extract feature matrix.

    Churn = customer hasn't purchased in CHURN_THRESHOLD_DAYS days.
    Recency is days since last purchase — higher = more likely churned.
    """
    try:
        df = df.copy()

        # ── Churn label ───────────────────────────────────────────────────────
        df["Churned"] = (df["Recency"] > CHURN_THRESHOLD_DAYS).astype(int)

        churn_rate = df["Churned"].mean()
        logging.info(
            f"Churn label defined — "
            f"churned: {df['Churned'].sum():,} | "
            f"active: {(df['Churned']==0).sum():,} | "
            f"rate: {churn_rate*100:.1f}%"
        )

        # ── Feature matrix ────────────────────────────────────────────────────
        # Guard: use only columns that exist in df
        available = [c for c in feature_columns if c in df.columns]
        missing   = [c for c in feature_columns if c not in df.columns]
        if missing:
            logging.warning(f"Feature columns not found in data: {missing}")

        X = df[available].fillna(0)
        y = df["Churned"]

        logging.info(
            f"Feature matrix — "
            f"shape: {X.shape} | "
            f"columns: {available}"
        )
        return X, y

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 4: Fit XGBoost ──────────────────────────────────────────────────────

def _fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    churn_cfg,
) -> xgb.XGBClassifier:
    """
    Train XGBoost with class imbalance correction via scale_pos_weight.
    Uses early stopping on the test set eval metric.
    """
    try:
        # Handle class imbalance — ratio of negatives to positives
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logging.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

        model = xgb.XGBClassifier(
            n_estimators      = churn_cfg.n_estimators,
            max_depth         = churn_cfg.max_depth,
            learning_rate     = churn_cfg.learning_rate,
            subsample         = churn_cfg.subsample,
            colsample_bytree  = churn_cfg.colsample_bytree,
            scale_pos_weight  = scale_pos_weight,
            eval_metric       = "auc",
            early_stopping_rounds = 20,
            random_state      = churn_cfg.random_state,
            verbosity         = 0,
        )

        model.fit(
            X_train, y_train,
            eval_set        = [(X_test, y_test)],
            verbose         = False,
        )

        logging.info(
            f"XGBoost fitted — "
            f"best iteration: {model.best_iteration} | "
            f"best AUC: {model.best_score:.4f}"
        )

        # ── Cross-validation on training set ──────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            xgb.XGBClassifier(
                n_estimators     = model.best_iteration,
                max_depth        = churn_cfg.max_depth,
                learning_rate    = churn_cfg.learning_rate,
                subsample        = churn_cfg.subsample,
                colsample_bytree = churn_cfg.colsample_bytree,
                scale_pos_weight = scale_pos_weight,
                random_state     = churn_cfg.random_state,
                verbosity        = 0,
            ),
            X_train, y_train,
            cv      = cv,
            scoring = "roc_auc",
        )
        logging.info(
            f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        return model

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 5: Evaluate ─────────────────────────────────────────────────────────

def _evaluate(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    try:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc  = average_precision_score(y_test, y_proba)
        cm      = confusion_matrix(y_test, y_pred)
        report  = classification_report(y_test, y_pred, output_dict=True)

        tn, fp, fn, tp = cm.ravel()

        logging.info(f"ROC-AUC  : {roc_auc:.4f}")
        logging.info(f"PR-AUC   : {pr_auc:.4f}")
        logging.info(f"Precision: {report['1']['precision']:.4f}")
        logging.info(f"Recall   : {report['1']['recall']:.4f}")
        logging.info(f"F1-score : {report['1']['f1-score']:.4f}")
        logging.info(f"Confusion matrix — TP:{tp} FP:{fp} TN:{tn} FN:{fn}")

        return {
            "roc_auc"   : roc_auc,
            "pr_auc"    : pr_auc,
            "precision" : report["1"]["precision"],
            "recall"    : report["1"]["recall"],
            "f1"        : report["1"]["f1-score"],
            "accuracy"  : report["accuracy"],
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 6a: SHAP values ─────────────────────────────────────────────────────

def _compute_shap(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
) -> np.ndarray:
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Log top 5 most important features by mean absolute SHAP
        mean_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X_test.columns,
        ).sort_values(ascending=False)

        logging.info("Top 5 features by mean |SHAP|:")
        for feat, val in mean_shap.head(5).items():
            logging.info(f"  {feat}: {val:.4f}")

        return shap_values

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 6b: Log to MLflow ───────────────────────────────────────────────────

def _log_mlflow(churn_cfg, metrics: dict, shap_values, X_test) -> None:
    try:
        mlflow.log_params({
            "n_estimators"     : churn_cfg.n_estimators,
            "max_depth"        : churn_cfg.max_depth,
            "learning_rate"    : churn_cfg.learning_rate,
            "subsample"        : churn_cfg.subsample,
            "colsample_bytree" : churn_cfg.colsample_bytree,
            "churn_threshold"  : CHURN_THRESHOLD_DAYS,
            "test_size"        : churn_cfg.test_size,
            "feature_count"    : len(churn_cfg.feature_columns),
        })

        mlflow.log_metrics({
            "roc_auc"   : round(metrics["roc_auc"],   4),
            "pr_auc"    : round(metrics["pr_auc"],    4),
            "precision" : round(metrics["precision"], 4),
            "recall"    : round(metrics["recall"],    4),
            "f1"        : round(metrics["f1"],        4),
            "accuracy"  : round(metrics["accuracy"],  4),
            "tp"        : metrics["tp"],
            "fp"        : metrics["fp"],
            "tn"        : metrics["tn"],
            "fn"        : metrics["fn"],
        })

        # Log mean SHAP importances as metrics
        mean_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X_test.columns,
        )
        for feat, val in mean_shap.items():
            mlflow.log_metric(f"shap_{feat}", round(float(val), 6))

        logging.info("MLflow logging complete ✓")

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 6c: Save artifact ───────────────────────────────────────────────────

def _save_artifact(model: xgb.XGBClassifier) -> None:
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, CHURN_MODEL_ARTIFACT)
        mlflow.xgboost.log_model(xgb_model=model,name="churn_xgb_model")
        logging.info(f"Churn model saved → {CHURN_MODEL_ARTIFACT}")

    except Exception as e:
        raise CSRException(e, sys)


# ─── Step 6d: Save predictions to Postgres ────────────────────────────────────

def _save_predictions(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    engine,
) -> None:
    """
    Write churn probability and risk tier per customer to Postgres.
    """
    try:
        preds = pd.DataFrame({
            COL_CUSTOMER_ID : df[COL_CUSTOMER_ID].values,
            "ChurnProb"     : model.predict_proba(X)[:, 1].round(4),
            "Churned"       : model.predict(X),
        })

        preds["ChurnRisk"] = pd.cut(
            preds["ChurnProb"],
            bins   = CHURN_RISK_BINS,
            labels = CHURN_RISK_LABELS,
        ).astype(str)

        full_table = f"{DB_SCHEMA}.{TABLE_CHURN_PREDICTIONS}"
        preds.to_sql(
            name      = TABLE_CHURN_PREDICTIONS,
            con       = engine,
            schema    = DB_SCHEMA,
            if_exists = "replace",
            index     = False,
            chunksize = DB_INSERT_CHUNKSIZE,
            method    = "multi",
        )

        # Risk tier breakdown
        risk_dist = preds["ChurnRisk"].value_counts()
        for tier, count in risk_dist.items():
            logging.info(
                f"  {tier} risk: {count:,} customers "
                f"({count/len(preds)*100:.1f}%)"
            )

        logging.info(
            f"Churn predictions saved ✓ — "
            f"{len(preds):,} rows → {full_table}"
        )

    except Exception as e:
        raise CSRException(e, sys)


if __name__ == "__main__":
    run_churn_pipeline()