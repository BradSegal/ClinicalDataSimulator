import gc
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from memory_profiler import memory_usage

from src.simulator import DataGenerationConfig, SyntheticDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScalabilityAnalysis:
    """Framework for analyzing computational scalability of the data generator."""

    def __init__(self, output_dir: str = "scalability_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Define complexity configurations
        self.complexity_configs = {
            'Low': {
                'n_samples': 1000,
                'n_sites': 3,
                'n_predictive_continuous': 5,
                'n_predictive_categorical': 5,
                'n_noise_continuous': 2,
                'n_noise_categorical': 2,
                'max_interaction_order': 2,
                'include_interactions': True,
                'include_nonlinear': False
            },
            'Medium': {
                'n_samples': 5000,
                'n_sites': 5,
                'n_predictive_continuous': 10,
                'n_predictive_categorical': 10,
                'n_noise_continuous': 5,
                'n_noise_categorical': 5,
                'max_interaction_order': 3,
                'include_interactions': True,
                'include_nonlinear': True
            },
            'High': {
                'n_samples': 10000,
                'n_sites': 10,
                'n_predictive_continuous': 20,
                'n_predictive_categorical': 20,
                'n_noise_continuous': 10,
                'n_noise_categorical': 10,
                'max_interaction_order': 4,
                'include_interactions': True,
                'include_nonlinear': True
            },
            'Very High': {
                'n_samples': 100000,
                'n_sites': 20,
                'n_predictive_continuous': 20,
                'n_predictive_categorical': 20,
                'n_noise_continuous': 10,
                'n_noise_categorical': 10,
                'max_interaction_order': 5,
                'include_interactions': True,
                'include_nonlinear': True
            }
        }

        self.results = []
        self.n_repetitions = 3  # Number of repetitions for each configuration

    def create_config(self, complexity_level: str) -> DataGenerationConfig:
        """Create configuration for specified complexity level."""
        params = self.complexity_configs[complexity_level]
        return DataGenerationConfig(**params)

    def measure_performance(self, config: DataGenerationConfig) -> Dict:
        """Measure time and memory performance for a single configuration."""
        gc.collect()  # Force garbage collection before measurement

        # Measure time and memory
        start_time = time.time()
        mem_usage = memory_usage((SyntheticDataGenerator(config).generate,), max_usage=True)

        # Generate data for additional metrics
        generator = SyntheticDataGenerator(config)
        data, relationships = generator.generate()
        runtime = time.time() - start_time

        # Calculate feature and interaction metrics
        total_features = (config.n_predictive_continuous + config.n_predictive_categorical +
                          config.n_noise_continuous + config.n_noise_categorical)

        n_interactions = len(relationships.get('interactions', [])) if config.include_interactions else 0

        return {
            'runtime': runtime,
            'memory': mem_usage,
            'feature_count': total_features,
            'interaction_order': config.max_interaction_order if config.include_interactions else 0,
            'n_interactions': n_interactions,
            'data_shape': data.shape
        }

    def run_analysis(self):
        """Execute scalability analysis across all configurations."""
        for complexity in self.complexity_configs.keys():
            logger.info(f"Analyzing {complexity} complexity configuration")

            config = self.create_config(complexity)

            # Run multiple repetitions
            for rep in range(self.n_repetitions):
                logger.info(f"Repetition {rep + 1}/{self.n_repetitions}")

                # Set different random seed for each repetition
                config.random_state = rep
                metrics = self.measure_performance(config)

                self.results.append({
                    'complexity': complexity,
                    'repetition': rep,
                    **metrics,
                    **asdict(config)
                })

        self.generate_analysis()

    def generate_latex_table(self) -> str:
        """Generate LaTeX table with scalability metrics."""
        results_df = pd.DataFrame(self.results)

        # Calculate mean metrics for each complexity level
        summary = results_df.groupby('complexity').agg({
            'runtime': ['mean', 'std'],
            'memory': ['mean', 'std'],
            'feature_count': 'first',
            'interaction_order': 'first'
        }).round(2)

        # Generate LaTeX table
        latex_rows = []
        for complexity in ['Low', 'Medium', 'High']:
            metrics = summary.loc[complexity]
            row = (f"{complexity} & "
                   f"{metrics['runtime']['mean']:.2f} ± {metrics['runtime']['std']:.2f} & "
                   f"{metrics['memory']['mean']:.1f} ± {metrics['memory']['std']:.1f} & "
                   f"{int(metrics['feature_count']['first'])} & "
                   f"{int(metrics['interaction_order']['first'])} \\\\")
            latex_rows.append(row)

        table = "\\begin{table}[h]\n\\caption{Scalability Performance Metrics}\n"
        table += "\\label{tab:scalability}\n\\begin{center}\n"
        table += "\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n"
        table += "Complexity & Time & Memory & Feature & Interaction \\\\\n"
        table += "Level & (s) & (MB) & Count & Order \\\\\n\\hline\n"
        table += "\n\\hline\n".join(latex_rows)
        table += "\n\\hline\n\\end{tabular}\n\\end{center}\n\\end{table}"

        # Save table to file
        with open(self.output_dir / 'scalability_table.tex', 'w') as f:
            f.write(table)

        return table

    def plot_scalability_metrics(self):
        """Create visualizations of scalability metrics."""
        results_df = pd.DataFrame(self.results)

        # Set style
        # plt.style.use('seaborn-whitegrid')

        # Time and Memory scaling plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Runtime plot
        sns.boxplot(data=results_df, x='complexity', y='runtime', ax=ax1)
        ax1.set_title('Runtime Scaling')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Complexity Level')

        # Memory plot
        sns.boxplot(data=results_df, x='complexity', y='memory', ax=ax2)
        ax2.set_title('Memory Usage Scaling')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_xlabel('Complexity Level')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'scalability_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Feature count vs Performance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=results_df, x='feature_count', y='runtime',
                        hue='complexity', style='complexity', s=100)
        ax.set_title('Runtime vs Feature Count')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Time (seconds)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'runtime_vs_features.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_analysis(self):
        """Generate comprehensive analysis outputs."""
        logger.info("Generating LaTeX table...")
        self.generate_latex_table()

        logger.info("Creating visualization plots...")
        self.plot_scalability_metrics()

        # Save detailed results
        pd.DataFrame(self.results).to_csv(self.output_dir / 'detailed_results.csv', index=False)


def main():
    """Execute scalability analysis."""
    logger.info("Starting scalability analysis")
    analysis = ScalabilityAnalysis()
    analysis.run_analysis()
    logger.info("Completed scalability analysis")


if __name__ == "__main__":
    main()
