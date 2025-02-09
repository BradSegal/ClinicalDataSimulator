import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from scipy import stats
from scipy.stats import linregress

from src.simulator import DataGenerationConfig, SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


COLOR_SCHEME = {
    'small': '#1f77b4',  # blue
    'medium': '#2ca02c',  # green
    'large': '#d62728',  # red
    'grid': '#E0E0E0',   # light grey for grids
    'identity': '#404040' # dark grey for identity lines
}


class StatisticalMetrics:
    """Statistical analysis utilities for validation metrics."""

    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean using t-distribution."""
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
        return ci[0], ci[1]

    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, statistic: callable,
                                      n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for arbitrary statistics."""
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(sample))
        return np.percentile(bootstrap_stats, [(1 - confidence) * 100 / 2, (1 + confidence) * 100 / 2])

    @staticmethod
    def effect_size_cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size between two groups."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se if pooled_se != 0 else 0


class SimulatorValidation:
    """Validation framework for clinical data simulator."""

    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.test_configs = [
            {
                'name': 'small',
                'n_samples': 1000,
                'n_sites': 3,
                'n_predictive': 10,
                'missing_rates': [0.0, 0.1, 0.2]
            },
            {
                'name': 'medium',
                'n_samples': 5000,
                'n_sites': 5,
                'n_predictive': 20,
                'missing_rates': [0.0, 0.1, 0.2]
            },
            {
                'name': 'large',
                'n_samples': 10000,
                'n_sites': 10,
                'n_predictive': 30,
                'missing_rates': [0.0, 0.1, 0.2]
            }
        ]

        self.results = []

    @staticmethod
    def create_config(params: Dict, missing_rate: float) -> DataGenerationConfig:
        """Initialize simulator configuration with specified parameters."""
        n_continuous = params['n_predictive'] // 2
        n_categorical = params['n_predictive'] - n_continuous

        return DataGenerationConfig(
            n_samples=params['n_samples'],
            n_sites=params['n_sites'],
            n_predictive_continuous=n_continuous,
            n_predictive_categorical=n_categorical,
            include_interactions=True,
            include_nonlinear=True,
            missing_rate=missing_rate,
            random_state=42
        )

    @staticmethod
    def analyse_effect_recovery(data: pd.DataFrame, relationships: Dict) -> Dict:
        """analyse correlation between true and observed feature effects."""
        metrics = {}
        true_effects = []
        observed_effects = []
        valid_features = []

        for feature, true_coef in relationships['continuous_coefficients'].items():
            valid_data = data[[feature, 'outcome']].dropna()
            if len(valid_data) > 10:  # Minimum sample size for correlation
                try:
                    obs_coef = valid_data[feature].corr(valid_data['outcome'])
                    if not pd.isna(obs_coef):
                        true_effects.append(true_coef)
                        observed_effects.append(obs_coef)
                        valid_features.append(feature)
                except Exception as e:
                    logger.warning(f"Correlation calculation failed for {feature}: {e}")

        if valid_features:
            slope, intercept, r_value, p_value, std_err = linregress(true_effects, observed_effects)
            metrics.update({
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'slope': slope,
                'intercept': intercept,
                'std_err': std_err,
                'true_effects': true_effects,
                'observed_effects': observed_effects,
                'features': valid_features
            })

        return metrics

    @staticmethod
    def analyse_site_variation(data: pd.DataFrame, relationships: Dict) -> Dict:
        """analyse variation in outcomes across clinical sites."""
        site_stats = data.groupby('site', observed=False)['outcome'].agg(['mean', 'std', 'count'])
        target_prevs = relationships['site_effects']

        site_comparisons = []
        for site in site_stats.index:
            achieved = site_stats.loc[site, 'mean']
            target = target_prevs[site]
            n_samples = site_stats.loc[site, 'count']
            std_err = site_stats.loc[site, 'std'] / np.sqrt(n_samples)

            site_comparisons.append({
                'site': site,
                'target': target,
                'achieved': achieved,
                'difference': achieved - target,
                'ci_lower': achieved - 1.96 * std_err,
                'ci_upper': achieved + 1.96 * std_err,
                'n_samples': n_samples
            })

        comparison_df = pd.DataFrame(site_comparisons)

        return {
            'mean_absolute_error': abs(comparison_df['difference']).mean(),
            'within_ci': (comparison_df['target'] >= comparison_df['ci_lower']) &
                         (comparison_df['target'] <= comparison_df['ci_upper']),
            'site_comparison_df': comparison_df,
            'prevalence_range': comparison_df['achieved'].max() - comparison_df['achieved'].min()
        }

    def run_single_test(self, config: DataGenerationConfig) -> Dict:
        """Execute single validation test with specified configuration."""
        logger.info(f"Testing configuration: {config.n_samples} samples, {config.n_sites} sites")

        start_time = time.time()
        mem_usage = memory_usage((SyntheticDataGenerator(config).generate,), max_usage=True)

        generator = SyntheticDataGenerator(config)
        data, relationships = generator.generate()
        runtime = time.time() - start_time

        effect_metrics = self.analyse_effect_recovery(data, relationships)
        site_metrics = self.analyse_site_variation(data, relationships)

        return {
            'config': asdict(config),
            'runtime': runtime,
            'memory': mem_usage,
            'effect_metrics': effect_metrics,
            'site_metrics': site_metrics,
            'data_shape': data.shape
        }

    def run_validation(self):
        """Execute validation suite across all configurations."""
        for config_params in self.test_configs:
            logger.info(f"Validating {config_params['name']} configuration")

            for missing_rate in config_params['missing_rates']:
                config = self.create_config(config_params, missing_rate)
                result = self.run_single_test(config)
                result['config_name'] = config_params['name']
                result['missing_rate'] = missing_rate
                self.results.append(result)

        self.generate_validation_outputs()

    def generate_validation_outputs(self):
        """Generate validation summary tables and visualizations."""
        self._create_summary_table()
        self._plot_effect_analysis()
        self._plot_site_analysis()

    def _create_summary_table(self) -> pd.DataFrame:
        """Create summary statistics table for validation results."""
        summary_data = []

        for result in self.results:
            if 'effect_metrics' in result and result['effect_metrics']:
                summary_data.append({
                    'Configuration': f"{result['config_name']} (missing: {result['missing_rate'] * 100}%)",
                    'Samples': result['data_shape'][0],
                    'Runtime (s)': f"{result['runtime']:.2f}",
                    'Memory (MB)': f"{result['memory']:.1f}",
                    'Effect R²': f"{result['effect_metrics'].get('r_squared', 0):.3f}",
                    'Effect p-value': f"{result['effect_metrics'].get('p_value', 1):.3e}",
                    'Site MAE': f"{result['site_metrics']['mean_absolute_error']:.3f}",
                    'Sites Within CI': f"{result['site_metrics']['within_ci'].mean() * 100:.1f}%"
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'validation_summary.csv', index=False)
        return summary_df

    def _plot_effect_analysis(self):
        """Plot analysis of feature effect recovery."""
        plt.figure(figsize=(10, 8))

        # Plot only complete data results (missing_rate = 0)
        complete_results = [r for r in self.results if r['missing_rate'] == 0]

        config_colors = {
            'small': COLOR_SCHEME['small'],
            'medium': COLOR_SCHEME['medium'],
            'large': COLOR_SCHEME['large']
        }

        for result in complete_results:
            if 'effect_metrics' not in result or not result['effect_metrics']:
                continue

            metrics = result['effect_metrics']
            true_effects = metrics['true_effects']
            observed_effects = metrics['observed_effects']
            config_name = result['config_name']

            # Scatter plot
            plt.scatter(
                true_effects, observed_effects,
                alpha=0.6, color=config_colors[config_name],
                label=f"{result['config_name']} (n={result['config']['n_samples']}, R²={metrics['r_squared']:.3f})"
            )

            # Trend line
            z = np.polyfit(true_effects, observed_effects, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(true_effects), max(true_effects), 100)
            plt.plot(x_range, p(x_range), '--', color=config_colors[config_name], alpha=0.8)

        plt.plot([plt.xlim()[0], plt.xlim()[1]],
                 [plt.xlim()[0], plt.xlim()[1]],
                 'k--', alpha=0.5, label='Perfect Recovery')

        plt.xlabel('True Effect Size')
        plt.ylabel('Observed Effect Size')
        plt.title('Feature Effect Recovery')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'effect_recovery.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_site_analysis(self):
        """Plot analysis of clinical site prevalence variations."""
        plt.figure(figsize=(10, 8))

        # Plot only complete data results
        complete_results = [r for r in self.results if r['missing_rate'] == 0]
        config_colors = {
            'small': COLOR_SCHEME['small'],
            'medium': COLOR_SCHEME['medium'],
            'large': COLOR_SCHEME['large']
        }
        for result in complete_results:
            site_df = result['site_metrics']['site_comparison_df']
            config_name = result['config_name']

            plt.errorbar(site_df['target'], site_df['achieved'],
                         yerr=[site_df['achieved'] - site_df['ci_lower'],
                               site_df['ci_upper'] - site_df['achieved']],
                         fmt='o', color=config_colors[config_name], alpha=0.6,
                         label=f"{result['config_name']} (n={result['config']['n_samples']})")

            # Add trend line
            z = np.polyfit(site_df['target'], site_df['achieved'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(site_df['target']), max(site_df['target']), 100)
            plt.plot(x_range, p(x_range), '--', color=config_colors[config_name], alpha=0.8)

        # Reference line
        min_val = plt.xlim()[0]
        max_val = plt.xlim()[1]
        plt.plot([min_val, max_val], [min_val, max_val],
                 'k--', alpha=0.5, label='Target')

        plt.xlabel('Target Site Prevalence')
        plt.ylabel('Achieved Site Prevalence')
        plt.title('Site-Specific Outcome Prevalence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.output_dir / 'site_variation.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Execute validation suite."""
    validation = SimulatorValidation()
    validation.run_validation()


if __name__ == "__main__":
    main()
