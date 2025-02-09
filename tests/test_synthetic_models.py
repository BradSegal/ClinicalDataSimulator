import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.simulator import SyntheticDataGenerator, DataGenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonConfig:
    """Configuration for model comparison experiment."""
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: Optional[int] = 42
    n_trials: int = 5
    n_bootstrap: int = 1000  # for confidence intervals


class FeatureImportanceComparator:
    """Compare feature importance between models and ground truth."""

    @staticmethod
    def normalize_importance(importance_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize feature importance scores to sum to 1."""
        total = sum(abs(v) for v in importance_dict.values())
        return {k: abs(v) / total for k, v in importance_dict.items()}

    @staticmethod
    def get_model_importance(model, feature_names: List[str]) -> Dict[str, float]:
        """Extract and normalize feature importance from trained model."""
        if hasattr(model, 'coef_'):
            importance = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            raise ValueError("Model does not provide feature importance")

        return dict(zip(feature_names, importance))

    @staticmethod
    def calculate_importance_correlation(true_imp: Dict[str, float],
                                         model_imp: Dict[str, float]) -> float:
        """Calculate correlation between true and model-derived feature importance."""
        common_features = set(true_imp.keys()) & set(model_imp.keys())
        if not common_features:
            return 0.0

        true_values = [true_imp[f] for f in common_features]
        model_values = [model_imp[f] for f in common_features]

        return np.corrcoef(true_values, model_values)[0, 1]


class ModelEvaluator:
    """Evaluate and compare multiple classification models."""

    def __init__(self, config: ModelComparisonConfig):
        """Initialize model evaluator with configuration."""
        self.config = config
        self.models = {
            'logistic': LogisticRegression(random_state=config.random_state),
            'random_forest': RandomForestClassifier(random_state=config.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=config.random_state),
            'xgboost': xgb.XGBClassifier(random_state=config.random_state)
        }

    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables."""
        # Exclude metadata columns but keep relevant categorical columns
        feature_cols = [col for col in data.columns
                        if col not in ['outcome', 'patient_id', 'timepoint']]

        X = data[feature_cols].copy()

        # Identify and process categorical columns
        categorical_cols = ['sex', 'site', 'age_group'] + [col for col in X.columns if col.startswith('pred_cat')]

        # Convert all categorical columns to strings to ensure uniform type
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)
                X[col] = X[col].astype('category')

        # Identify numeric columns (non-categorical)
        numeric_cols = [col for col in X.columns if col not in categorical_cols]

        # Ensure numeric columns are float
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        y = data['outcome']

        logger.info(f"Prepared {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
        logger.info(f"Categorical features: {categorical_cols}")

        return X, y

    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create preprocessing pipeline with proper handling of categorical variables."""
        # Identify numeric and categorical columns
        categorical_cols = ['sex', 'site', 'age_group'] + [col for col in X.columns if col.startswith('pred_cat')]
        numeric_cols = [col for col in X.columns if col not in categorical_cols]

        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', dtype=np.float64))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'  # Drop any columns not explicitly specified
        )

        # Log preprocessing info
        logger.info(f"Created preprocessing pipeline:")
        logger.info(f"  Numeric features: {len(numeric_cols)}")
        logger.info(f"  Categorical features: {len(categorical_cols)}")

        return preprocessor

    def calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive set of performance metrics."""
        # Find optimal threshold using precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        optimal_threshold = thresholds[np.argmax(f1_scores[:-1])]  # exclude last point

        # Generate predictions using optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'avg_precision': average_precision_score(y_true, y_pred_proba),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1': f1_score(y_true, y_pred),
            'optimal_threshold': optimal_threshold
        }

        return metrics

    def calculate_confidence_intervals(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Calculate 95% confidence intervals for metrics using bootstrap."""
        bootstrap_metrics = []

        for _ in range(self.config.n_bootstrap):
            # Bootstrap sample indices
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)

            # Calculate metrics for this bootstrap sample
            metrics = self.calculate_metrics(y_true[indices], y_pred_proba[indices])
            bootstrap_metrics.append(metrics)

        # Calculate confidence intervals for each metric
        ci_metrics = {}
        for metric in bootstrap_metrics[0].keys():
            values = [m[metric] for m in bootstrap_metrics]
            ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
            ci_metrics[f"{metric}_ci"] = (ci_lower, ci_upper)

        return ci_metrics

    def evaluate_model(self, model, X_train: pd.DataFrame, X_val: pd.DataFrame,
                       y_train: pd.Series, y_val: pd.Series) -> Tuple[Dict, Pipeline]:
        """Train and evaluate a single model."""
        # Create and fit preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(X_train)

        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(y_val, y_pred_proba)

        # Calculate confidence intervals
        ci_metrics = self.calculate_confidence_intervals(y_val.values, y_pred_proba)
        metrics.update(ci_metrics)

        return metrics, pipeline

    def run_comparison(self, data: pd.DataFrame) -> Dict:
        """Run complete model comparison experiment."""
        X, y = self.prepare_data(data)

        # Split data into train/val/test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=self.config.val_size,
            random_state=self.config.random_state
        )

        results = {}
        best_model = None
        best_score = -float('inf')

        # Evaluate each model
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            metrics, pipeline = self.evaluate_model(model, X_train, X_val, y_train, y_val)

            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = {'name': name, 'pipeline': pipeline}

            results[name] = metrics

        # Evaluate best model on test set
        if best_model:
            test_pred_proba = best_model['pipeline'].predict_proba(X_test)[:, 1]
            test_metrics = self.calculate_metrics(y_test, test_pred_proba)
            test_ci = self.calculate_confidence_intervals(y_test.values, test_pred_proba)

            results['best_model'] = {
                'name': best_model['name'],
                'pipeline': best_model['pipeline'],
                **test_metrics,
                **test_ci
            }

        return results


def format_results_table(results: Dict) -> pd.DataFrame:
    """Format model comparison results as a clean DataFrame with confidence intervals."""
    # Define metrics to include and their display names
    metrics_display = {
        'roc_auc': 'AUROC',
        'avg_precision': 'AP',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'ppv': 'PPV',
        'npv': 'NPV',
        'f1': 'F1',
        'optimal_threshold': 'Optimal Threshold'
    }

    # Initialize results table
    table_data = []

    # Process each model's results
    for model_name, metrics in results.items():
        if model_name != 'best_model':
            row = {'Model': model_name.replace('_', ' ').title()}

            # Add point estimates
            for metric, display_name in metrics_display.items():
                value = metrics[metric]
                ci_lower, ci_upper = metrics[f"{metric}_ci"]
                row[display_name] = f"{value:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"

            table_data.append(row)

    # Create DataFrame and set display format
    results_df = pd.DataFrame(table_data)
    results_df.set_index('Model', inplace=True)

    return results_df


def plot_model_comparison(results: Dict, output_dir: str = "results"):
    """Create visualization of model comparison results."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Prepare data for plotting
    model_names = []
    aurocs = []
    auroc_cis = []

    for model_name, metrics in results.items():
        if model_name != 'best_model':
            model_names.append(model_name.replace('_', ' ').title())
            aurocs.append(metrics['roc_auc'])
            auroc_cis.append(metrics['roc_auc_ci'])

    # Create AUROC comparison plot
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(model_names))

    plt.barh(y_pos, aurocs, align='center', alpha=0.8)

    # Add error bars for confidence intervals
    ci_errors = np.array([(high - low) / 2 for low, high in auroc_cis])
    plt.errorbar(aurocs, y_pos, xerr=ci_errors, fmt='none', color='black', capsize=5)

    plt.yticks(y_pos, model_names)
    plt.xlabel('AUROC')
    plt.title('Model Performance Comparison')

    # Add value labels
    for i, v in enumerate(aurocs):
        plt.text(v, i, f' {v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Run complete model comparison experiment."""
    # Create configurations
    data_config = DataGenerationConfig(
        n_samples=5000,
        n_predictive_continuous=10,
        n_predictive_categorical=10,
        include_interactions=True,
        include_nonlinear=True,
        missing_rate=0.1,
        random_state=42
    )

    model_config = ModelComparisonConfig()

    # Generate synthetic data
    logger.info("Generating synthetic data...")
    generator = SyntheticDataGenerator(data_config)
    data, relationships = generator.generate()

    # Run model comparison
    logger.info("Running model comparison...")
    evaluator = ModelEvaluator(model_config)
    results = evaluator.run_comparison(data)

    # Format and display results table
    results_table = format_results_table(results)
    print("\nModel Comparison Results:")
    print(results_table.to_string())

    # Print best model details
    print("\nBest Model Details:")
    print(f"Model: {results['best_model']['name']}")
    print("\nTest Set Performance:")
    for metric in ['roc_auc', 'avg_precision', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']:
        value = results['best_model'][metric]
        ci_lower, ci_upper = results['best_model'][f"{metric}_ci"]
        print(f"{metric}: {value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")

    # Generate visualization
    plot_model_comparison(results)

    # Compare with true feature importance
    true_importance = generator.get_feature_importance()
    model_importance = FeatureImportanceComparator.get_model_importance(
        results['best_model']['pipeline']['classifier'],
        data.drop(columns=['outcome']).columns
    )

    # Save results to file
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    results_table.to_csv(output_dir / "model_comparison_results.csv")
    true_importance.to_csv(output_dir / "true_feature_importance.csv")
    pd.Series(model_importance).to_csv(output_dir / "model_feature_importance.csv")



if __name__ == "__main__":
    main()