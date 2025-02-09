import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from src.simulator import DataGenerationConfig, SyntheticDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define color scheme
COLOR_SCHEME = {
    'small': '#1f77b4',  # blue
    'medium': '#2ca02c',  # green
    'large': '#d62728',  # red
    'grid': '#E0E0E0',  # light grey for grids
    'identity': '#404040'  # dark grey for identity lines
}


class EffectRecoveryValidation:
    """Validation framework for effect size recovery analysis using logistic regression."""

    def __init__(self, output_dir: str = "effect_recovery_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Define test configurations
        self.sample_sizes = [1000, 5000, 10000]
        self.effect_sizes = [0.1, 0.3, 0.5, 0.8, 1.0]
        self.n_repetitions = 5
        self.results = []

        # Set default plot style
        plt.style.use('default')  # Use matplotlib's default style
        sns.set_theme(style="whitegrid")  # Set seaborn style separately

    def create_config(self, n_samples: int, effect_size: float) -> DataGenerationConfig:
        """Initialize simulator configuration for effect size testing."""
        return DataGenerationConfig(
            n_samples=n_samples,
            n_sites=3,
            n_predictive_continuous=5,
            n_predictive_categorical=5,
            include_interactions=False,
            include_nonlinear=False,
            missing_rate=0.0,
            random_state=None,
            feature_effect_size=effect_size
        )

    def analyze_effect_recovery(self, data: pd.DataFrame, relationships: Dict) -> Dict:
        """Analyze recovery of feature effects using logistic regression."""
        results = {
            'continuous': self._analyze_continuous_effects(data, relationships),
            'categorical': self._analyze_categorical_effects(data, relationships)
        }

        # Calculate overall metrics
        all_true_effects = results['continuous']['true_effects'] + results['categorical']['true_effects']
        all_observed_effects = results['continuous']['observed_effects'] + results['categorical']['observed_effects']

        overall_metrics = {
            'mae': np.mean(np.abs(np.array(all_true_effects) - np.array(all_observed_effects))),
            'relative_error': np.mean(np.abs(np.array(all_true_effects) - np.array(all_observed_effects)) /
                                      np.abs(np.array(all_true_effects)))
        }

        return {**results, 'overall_metrics': overall_metrics}

    def _analyze_continuous_effects(self, data: pd.DataFrame, relationships: Dict) -> Dict:
        """Analyze continuous feature effects using logistic regression."""
        return self._analyze_effects(data, relationships, 'continuous_coefficients')

    def _analyze_categorical_effects(self, data: pd.DataFrame, relationships: Dict) -> Dict:
        """Analyze categorical feature effects using logistic regression."""
        return self._analyze_effects(data, relationships, 'categorical_coefficients')

    def _analyze_effects(self, data: pd.DataFrame, relationships: Dict, coef_type: str) -> Dict:
        """Generic effect analysis for both continuous and categorical features."""
        true_effects = []
        observed_effects = []
        confidence_intervals = []
        p_values = []
        features = []

        for feature, true_coef in relationships[coef_type].items():
            # Prepare data
            X = (data[feature].values if coef_type == 'continuous_coefficients'
                 else data[feature].cat.codes.values).reshape(-1, 1)
            y = data['outcome'].values

            # Fit logistic regression
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            obs_coef = model.coef_[0][0]

            # Calculate confidence interval and p-value
            X_design = np.column_stack([np.ones(len(X)), X])
            pred_probs = model.predict_proba(X)[:, 1]
            W = np.diag(pred_probs * (1 - pred_probs))

            try:
                var_coef = np.linalg.inv(X_design.T @ W @ X_design)[1, 1]
                std_err = np.sqrt(var_coef)
                ci = (obs_coef - 1.96 * std_err, obs_coef + 1.96 * std_err)
                p_value = norm.sf(abs(obs_coef / std_err)) * 2
            except np.linalg.LinAlgError:
                ci = (np.nan, np.nan)
                p_value = np.nan

            features.append(feature)
            true_effects.append(true_coef)
            observed_effects.append(obs_coef)
            confidence_intervals.append(ci)
            p_values.append(p_value)

        # Calculate metrics
        mae = np.mean(np.abs(np.array(true_effects) - np.array(observed_effects)))
        relative_error = np.mean(np.abs(np.array(true_effects) - np.array(observed_effects)) /
                                 np.abs(np.array(true_effects)))

        return {
            'features': features,
            'true_effects': true_effects,
            'observed_effects': observed_effects,
            'confidence_intervals': confidence_intervals,
            'p_values': p_values,
            'mae': mae,
            'relative_error': relative_error
        }

    def run_single_test(self, config: DataGenerationConfig) -> Dict:
        """Execute single validation test."""
        start_time = time.time()
        generator = SyntheticDataGenerator(config)
        data, relationships = generator.generate()
        runtime = time.time() - start_time

        effect_metrics = self.analyze_effect_recovery(data, relationships)

        return {
            'config': asdict(config),
            'runtime': runtime,
            'effect_metrics': effect_metrics
        }

    def run_validation(self):
        """Execute validation across all configurations."""
        for n_samples in self.sample_sizes:
            for effect_size in self.effect_sizes:
                logger.info(f"Testing n_samples={n_samples}, effect_size={effect_size}")

                for rep in range(self.n_repetitions):
                    config = self.create_config(n_samples, effect_size)
                    config.random_state = rep
                    result = self.run_single_test(config)
                    result.update({
                        'n_samples': n_samples,
                        'effect_size': effect_size,
                        'repetition': rep
                    })
                    self.results.append(result)

        self.generate_analysis()

    def _plot_logistic_recovery_scatter(self):
        """Plot logistic regression recovery colored by sample size."""
        plt.figure(figsize=(10, 8))

        # Map sample sizes to colors
        size_colors = {
            1000: COLOR_SCHEME['small'],
            5000: COLOR_SCHEME['medium'],
            10000: COLOR_SCHEME['large']
        }

        # Collect data
        all_true = []
        all_logistic = []
        all_sample_sizes = []
        feature_types = []

        for result in self.results:
            # Continuous features
            all_true.extend(result['effect_metrics']['continuous']['true_effects'])
            all_logistic.extend(result['effect_metrics']['continuous']['observed_effects'])
            all_sample_sizes.extend([result['n_samples']] *
                                    len(result['effect_metrics']['continuous']['true_effects']))
            feature_types.extend(['continuous'] *
                                 len(result['effect_metrics']['continuous']['true_effects']))

            # Categorical features
            all_true.extend(result['effect_metrics']['categorical']['true_effects'])
            all_logistic.extend(result['effect_metrics']['categorical']['observed_effects'])
            all_sample_sizes.extend([result['n_samples']] *
                                    len(result['effect_metrics']['categorical']['true_effects']))
            feature_types.extend(['categorical'] *
                                 len(result['effect_metrics']['categorical']['true_effects']))

        # Plot points and calculate R² for each sample size
        for sample_size in self.sample_sizes:
            mask = np.array(all_sample_sizes) == sample_size
            true_subset = np.array(all_true)[mask]
            logistic_subset = np.array(all_logistic)[mask]

            # Calculate R² for this sample size
            correlation_matrix = np.corrcoef(true_subset, logistic_subset)
            r2 = correlation_matrix[0, 1] ** 2

            plt.scatter(true_subset,
                        logistic_subset,
                        alpha=0.6,
                        label=f'n={sample_size:,} (R²={r2:.3f})',
                        color=size_colors[sample_size])

        # Add identity line
        min_val = min(min(all_true), min(all_logistic))
        max_val = max(max(all_true), max(all_logistic))
        plt.plot([min_val, max_val], [min_val, max_val], '--',
                 color=COLOR_SCHEME['identity'], alpha=0.5)

        plt.xlabel('True Effects')
        plt.ylabel('Logistic Regression Estimates')
        plt.legend()
        plt.grid(True, alpha=0.3, color=COLOR_SCHEME['grid'])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'logistic_recovery_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_mae_by_effect_size(self):
        """Plot MAE comparison across effect sizes for each feature type."""
        # Set color palette for this plot
        current_palette = [COLOR_SCHEME['small'], COLOR_SCHEME['medium'], COLOR_SCHEME['large']]

        for feature_type in ['continuous', 'categorical']:
            plt.figure(figsize=(10, 6))

            error_data = []
            for sample_size in self.sample_sizes:
                for effect_size in self.effect_sizes:
                    results = [r for r in self.results
                               if r['n_samples'] == sample_size and
                               r['effect_size'] == effect_size]

                    for result in results:
                        error_data.append({
                            'Effect Size': effect_size,
                            'MAE': result['effect_metrics'][feature_type]['mae'],
                            'Sample Size': f'n={sample_size:,}'
                        })

            error_df = pd.DataFrame(error_data)
            sns.boxplot(data=error_df, x='Effect Size', y='MAE', hue='Sample Size',
                        palette=current_palette)

            plt.title(f'MAE by Effect Size for {feature_type.title()} Features')
            plt.grid(True, alpha=0.3, color=COLOR_SCHEME['grid'])

            plt.tight_layout()
            plt.savefig(self.output_dir / f'mae_{feature_type}_features.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_confidence_intervals(self):
        """Plot confidence intervals for effect estimates."""
        max_sample_size = max(self.sample_sizes)
        max_effect_size = max(self.effect_sizes)
        selected_results = [r for r in self.results
                            if r['n_samples'] == max_sample_size and
                            r['effect_size'] == max_effect_size]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        def plot_effects(ax, feature_type, title):
            features = selected_results[0]['effect_metrics'][feature_type]['features']
            x_pos = np.arange(len(features))

            # Collect effects across repetitions
            all_true_effects = np.array([r['effect_metrics'][feature_type]['true_effects']
                                         for r in selected_results])
            all_obs_effects = np.array([r['effect_metrics'][feature_type]['observed_effects']
                                        for r in selected_results])

            # Calculate means and CIs
            true_means = np.mean(all_true_effects, axis=0)
            obs_means = np.mean(all_obs_effects, axis=0)
            obs_stds = np.std(all_obs_effects, axis=0)
            yerr = 1.96 * obs_stds

            # Plot
            ax.errorbar(x_pos, obs_means, yerr=yerr, fmt='o',
                        label='Observed Effects', color=COLOR_SCHEME['medium'])
            ax.plot(x_pos, true_means, '*', label='True Effects',
                    color=COLOR_SCHEME['large'], markersize=10)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(features, rotation=45)
            ax.set_title(f'{title}\nwith 95% Confidence Intervals')
            ax.legend()
            ax.grid(True, alpha=0.3, color=COLOR_SCHEME['grid'])

        plot_effects(ax1, 'continuous', 'Continuous Feature Effects')
        plot_effects(ax2, 'categorical', 'Categorical Feature Effects')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_summary_table(self) -> pd.DataFrame:
        """Create summary statistics table."""
        summary_data = []

        for n_samples in self.sample_sizes:
            for effect_size in self.effect_sizes:
                relevant_results = [r for r in self.results
                                    if r['n_samples'] == n_samples and
                                    r['effect_size'] == effect_size]

                for feature_type in ['continuous', 'categorical']:
                    mae_values = [r['effect_metrics'][feature_type]['mae']
                                  for r in relevant_results]
                    rel_error_values = [r['effect_metrics'][feature_type]['relative_error']
                                        for r in relevant_results]

                    summary_data.append({
                        'Sample Size': n_samples,
                        'Effect Size': effect_size,
                        'Feature Type': feature_type,
                        'Mean MAE': np.mean(mae_values),
                        'MAE Std': np.std(mae_values),
                        'Mean Relative Error': np.mean(rel_error_values),
                        'Relative Error Std': np.std(rel_error_values)
                    })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'effect_recovery_summary.csv', index=False)
        return summary_df

    def generate_analysis(self):
        """Generate comprehensive analysis of effect recovery."""
        logger.info("Generating summary table...")
        self._create_summary_table()

        logger.info("Creating logistic recovery scatter plot...")
        self._plot_logistic_recovery_scatter()

        logger.info("Creating MAE plots...")
        self._plot_mae_by_effect_size()

        logger.info("Creating confidence interval plots...")
        self._plot_confidence_intervals()


def main():
    """Execute effect recovery validation suite."""
    logger.info("Starting effect recovery validation")
    validation = EffectRecoveryValidation()
    validation.run_validation()
    logger.info("Completed effect recovery validation")


if __name__ == "__main__":
    main()