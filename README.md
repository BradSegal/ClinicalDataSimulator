# Clinical Data Simulator

## Overview
The Clinical Data Simulator is a research tool designed to generate synthetic medical data for testing and development purposes. This project aims to create realistic, privacy-compliant clinical datasets that maintain the statistical properties and relationships found in real medical data while ensuring no actual patient information is used.

## Features
- Generation of synthetic patient demographics
- Simulation of medical conditions and diagnoses
- Creation of temporal medical events
- Configurable data generation parameters
- Export capabilities to common healthcare data formats

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/BradSegal/ClinicalDataSimulator.git
cd ClinicalDataSimulator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from src.simulator import SyntheticDataGenerator, DataGenerationConfig

# Initialize with default configuration
config = DataGenerationConfig(
    n_samples=1000,
    n_sites=3,
    n_timepoints=1,
    random_state=42  # For reproducibility
)

# Create generator and generate data
generator = SyntheticDataGenerator(config)
data, relationships = generator.generate()

# Export the dataset
data.to_csv("synthetic_data.csv")
```

### Advanced Configuration
The simulator can be extensively customized using configuration parameters:

```python
config = DataGenerationConfig(
    n_samples=10000,
    n_sites=10,
    n_predictive_continuous=8,
    n_predictive_categorical=8,
    n_noise_continuous=8,
    n_noise_categorical=8,
    classification_noise=0.1,
    include_interactions=True,
    include_nonlinear=True,
    feature_effect_size=1.0,
    interaction_probability=0.3,
    interaction_effect_size=0.5,
    max_interaction_order=3,
    subgroup_effect_size=0.5,
    site_prevalence_range=(0.1, 0.4),
    missing_rate=0.1,
    random_state=42
)

generator = SyntheticDataGenerator(config)
data, relationships = generator.generate()

# Get feature importance analysis
feature_importance = generator.get_feature_importance()

# Get site-specific statistics
site_stats = generator.get_site_statistics()
```

## Data Structure
The generated data includes the following components:

### Main Dataset (`data` DataFrame)
- **Demographics**
  - Age (continuous)
  - Sex (categorical: M/F)
  - Age groups (categorical: 10-year bins)

- **Predictive Features**
  - Continuous predictive features (`pred_cont_0` through `pred_cont_n`)
  - Categorical predictive features (`pred_cat_0` through `pred_cat_n`)
  
- **Noise Features**
  - Continuous noise features (`noise_cont_0` through `noise_cont_n`)
  - Categorical noise features (`noise_cat_0` through `noise_cat_n`)

- **Site Information**
  - Site ID (categorical)
  - Site-specific effects

- **Outcome**
  - Binary classification outcome

### Relationships Object
Contains information about:
- Feature importance scores
- Interaction effects
- Subgroup effects
- Site-specific statistics
- Generated data relationships and parameters

The number of features in each category is controlled by the configuration parameters:
- `n_predictive_continuous`: Number of continuous predictive features
- `n_predictive_categorical`: Number of categorical predictive features
- `n_noise_continuous`: Number of continuous noise features
- `n_noise_categorical`: Number of categorical noise features

Missing data can be introduced using the `missing_rate` parameter in the configuration.

## Contributing
We welcome contributions to improve the Clinical Data Simulator. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
For academic use, please ensure you cite this work using the citation information below.

## Citation
If you use this simulator in your research, please cite:

```bibtex
@software{clinical_data_simulator,
    title = {Multi-Site Clinical Data Simulator},
    author = {Bradley Segal},
    year = {2024},
    url = {https://github.com/Brad Segal/ClinicalDataSimulator}
}
```

## Contact
For questions or support, please open an issue in the GitHub repository.
