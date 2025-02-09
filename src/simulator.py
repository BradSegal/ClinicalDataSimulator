import itertools

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List

from src.utility import logit, sigmoid


@dataclass
class DistributionConfig:
    """Configuration for feature distributions.

    Attributes:
        name: Name of the distribution ('normal', 'uniform', 'gamma', 'beta', etc.)
        params: Dictionary of parameters specific to the distribution
    """
    name: str
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataGenerationConfig:
    """Configuration parameters for synthetic data generation.

    Attributes:
        n_samples: Number of unique samples/patients
        n_sites: Number of clinical sites
        n_predictive_continuous: Number of continuous predictive features
        n_predictive_categorical: Number of categorical predictive features
        n_noise_continuous: Number of continuous noise features
        n_noise_categorical: Number of categorical noise features
        cat_min: Minimum number of categories for categorical features
        cat_max: Maximum number of categories for categorical features
        site_prevalence_range: Range of outcome prevalence across sites (min, max)
        include_interactions: Whether to include interaction effects
        include_nonlinear: Whether to include nonlinear transformations
        missing_rate: Proportion of values to set as missing
        n_timepoints: Number of timepoints per sample
        random_state: Random seed for reproducibility
        feature_effect_size: Standard deviation for feature coefficients
        temporal_effect_size: Size of temporal variations
        categorical_change_prob: Probability of categorical feature changes between timepoints
        interaction_probability: Probability of including an interaction between any two features
        interaction_effect_size: Size of interaction effects
        site_specific_effects: Whether to include site-specific effects
        max_interaction_order: Maximum number of features in an interaction
        subgroup_effect_size: Size of subgroup-specific effects
        feature_specific_subgroup_scale: Scaling factor for feature-specific subgroup effects
        hierarchical_effect_scale: Scaling factor for hierarchical effects
        age_ave: Average age for the population
        age_std: Standard deviation of age for the population
        male_prob: Probability of a sample being male. Considered binary between male & female sexes.
    """
    n_samples: int = 1000
    n_sites: int = 3
    n_predictive_continuous: int = 2
    n_predictive_categorical: int = 2
    n_noise_continuous: int = 2
    n_noise_categorical: int = 2
    cat_min: int = 2
    cat_max: int = 5
    site_prevalence_range: Tuple[float, float] = (0.2, 0.4)
    classification_noise: float = 0.05
    include_interactions: bool = False
    include_nonlinear: bool = False
    missing_rate: float = 0.0
    n_timepoints: int = 1
    random_state: Optional[int] = None
    feature_effect_size: float = 1.0
    temporal_effect_size: float = 0.1
    categorical_change_prob: float = 0.1
    interaction_probability: float = 0.3
    interaction_effect_size: float = 0.25
    site_specific_effects: bool = False
    max_interaction_order: int = 3
    subgroup_effect_size: float = 0.5
    feature_specific_subgroup_scale: float = 0.5
    hierarchical_effect_scale: float = 0.7
    age_ave: float = 50.0
    age_std: float = 15.0
    male_prob: float = 0.5
    continuous_distributions: Optional[List[DistributionConfig]] = None
    categorical_distributions: Optional[List[str]] = None  # 'uniform', 'binomial', 'geometric'
    default_continuous_dist: DistributionConfig = field(
        default_factory=lambda: DistributionConfig(name='normal', params={'loc': 0, 'scale': 1})
    )
    default_categorical_dist: str = 'uniform'

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive")
        if self.n_timepoints <= 0:
            raise ValueError("n_timepoints must be positive")
        if not 0 <= self.missing_rate <= 1:
            raise ValueError("missing_rate must be between 0 and 1")
        if not 0 < self.site_prevalence_range[0] < self.site_prevalence_range[1] < 1:
            raise ValueError("Invalid site_prevalence_range")
        if self.feature_effect_size <= 0:
            raise ValueError("feature_effect_size must be positive")
        if not 0 <= self.categorical_change_prob <= 1:
            raise ValueError("categorical_change_prob must be between 0 and 1")
        if not 0 <= self.interaction_probability <= 1:
            raise ValueError("interaction_probability must be between 0 and 1")
        if self.max_interaction_order < 1:
            raise ValueError("max_interaction_order must be at least 1")
        if self.subgroup_effect_size < 0:
            raise ValueError("subgroup_effect_size must be positive")
        if not (0 <= self.age_ave <= 100):
            raise ValueError("age_ave must be between 0 and 100")
        if self.age_std <= 0:
            raise ValueError("age_std must be positive")
        if not 0 <= self.male_prob <= 1:
            raise ValueError("male_prob must be between 0 and 1")


class FeatureGenerator:
    """Base class for generating synthetic features with configurable distributions."""

    # List of available distributions for random selection
    AVAILABLE_CONTINUOUS_DISTS = [
        ('normal', {'loc': 0, 'scale': 1}),
        ('uniform', {'low': -1, 'high': 1}),
        ('gamma', {'shape': 2, 'scale': 1}),
        ('beta', {'a': 2, 'b': 2}),
        ('exponential', {'scale': 1}),
        ('lognormal', {'mean': 0, 'sigma': 1})
    ]

    AVAILABLE_CATEGORICAL_DISTS = ['uniform', 'binomial', 'geometric']

    CONTINUOUS_DISTRIBUTIONS = {
        'normal': lambda rng, size, params: rng.normal(
            loc=params.get('loc', 0),
            scale=params.get('scale', 1),
            size=size
        ),
        'uniform': lambda rng, size, params: rng.uniform(
            low=params.get('low', -1),
            high=params.get('high', 1),
            size=size
        ),
        'gamma': lambda rng, size, params: rng.gamma(
            shape=params.get('shape', 2),
            scale=params.get('scale', 1),
            size=size
        ),
        'beta': lambda rng, size, params: rng.beta(
            a=params.get('a', 2),
            b=params.get('b', 2),
            size=size
        ),
        'exponential': lambda rng, size, params: rng.exponential(
            scale=params.get('scale', 1),
            size=size
        ),
        'lognormal': lambda rng, size, params: rng.lognormal(
            mean=params.get('mean', 0),
            sigma=params.get('sigma', 1),
            size=size
        )
    }

    CATEGORICAL_DISTRIBUTIONS = {
        'uniform': lambda rng, size, n_categories: rng.choice(
            range(n_categories),
            size=size,
            p=None
        ),
        'binomial': lambda rng, size, n_categories: rng.binomial(
            n=n_categories - 1,
            p=0.5,
            size=size
        ),
        'geometric': lambda rng, size, n_categories: np.minimum(
            rng.geometric(p=0.5, size=size) - 1,
            n_categories - 1
        )
    }

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    def _get_random_continuous_dist(self) -> DistributionConfig:
        """Randomly select a continuous distribution and its parameters."""
        dist_name, default_params = self.rng.choice(self.AVAILABLE_CONTINUOUS_DISTS)
        return DistributionConfig(name=dist_name, params=default_params)

    def _get_random_categorical_dist(self) -> str:
        """Randomly select a categorical distribution."""
        return self.rng.choice(self.AVAILABLE_CATEGORICAL_DISTS)

    def generate_continuous(
            self,
            n_samples: int,
            n_features: int,
            prefix: str = 'cont',
            n_timepoints: int = 1,
            distributions: Optional[List[DistributionConfig]] = None,
            default_dist: Optional[DistributionConfig] = None
    ) -> Dict[str, np.ndarray]:
        """Generate continuous features with specified distributions.

        Args:
            n_samples: Number of unique samples
            n_features: Number of features to generate
            prefix: Prefix for feature names
            n_timepoints: Number of timepoints per sample
            distributions: List of distribution configurations for each feature
            default_dist: Default distribution configuration if none specified
        """
        if distributions is None and default_dist is None:
            default_dist = DistributionConfig(name='normal', params={'loc': 0, 'scale': 1})

        features = {}
        total_samples = n_samples * n_timepoints

        for i in range(n_features):
            if distributions and i < len(distributions):
                dist_config = distributions[i]
                # Handle 'random' distribution selection
                if isinstance(dist_config, str) and dist_config.lower() == 'random':
                    dist_config = self._get_random_continuous_dist()
            else:
                dist_config = default_dist

            if dist_config.name not in self.CONTINUOUS_DISTRIBUTIONS:
                raise ValueError(f"Unsupported distribution: {dist_config.name}")

            values = self.CONTINUOUS_DISTRIBUTIONS[dist_config.name](
                self.rng,
                total_samples,
                dist_config.params
            )
            features[f'{prefix}_{i}'] = values

        return features

    def generate_categorical(
            self,
            n_samples: int,
            n_features: int,
            min_cat: int = 2,
            max_cat: int = 5,
            prefix: str = 'cat',
            n_timepoints: int = 1,
            distributions: Optional[List[str]] = None,
            default_dist: str = 'uniform'
    ) -> Dict[str, pd.Series]:
        """Generate categorical features with specified distributions.

        Args:
            n_samples: Number of unique samples
            n_features: Number of features to generate
            min_cat: Minimum number of categories
            max_cat: Maximum number of categories
            prefix: Prefix for feature names
            n_timepoints: Number of timepoints per sample
            distributions: List of distribution names for each feature
            default_dist: Default distribution if none specified
        """
        features = {}
        total_samples = n_samples * n_timepoints

        for i in range(n_features):
            n_categories = self.rng.randint(min_cat, max_cat + 1)
            if distributions and i < len(distributions):
                dist_name = distributions[i]
                # Handle 'random' distribution selection
                if dist_name.lower() == 'random':
                    dist_name = self._get_random_categorical_dist()
            else:
                dist_name = default_dist

            if dist_name not in self.CATEGORICAL_DISTRIBUTIONS:
                raise ValueError(f"Unsupported distribution: {dist_name}")

            values = self.CATEGORICAL_DISTRIBUTIONS[dist_name](
                self.rng,
                total_samples,
                n_categories
            )
            features[f'{prefix}_{i}'] = pd.Series(values).astype('category')

        return features


class TemporalFeatureGenerator(FeatureGenerator):
    """Generates features with temporal variations and configurable distributions."""

    def __init__(self, rng: np.random.RandomState, temporal_effect_size: float = 0.1,
                 categorical_change_prob: float = 0.1):
        if temporal_effect_size <= 0:
            raise ValueError("temporal_effect_size must be positive")
        if not 0 <= categorical_change_prob <= 1:
            raise ValueError("categorical_change_prob must be between 0 and 1")

        super().__init__(rng)
        self.temporal_effect_size = temporal_effect_size
        self.categorical_change_prob = categorical_change_prob

    def add_temporal_variation(self, base_values: np.ndarray,
                               n_timepoints: int,
                               dist_config: Optional[DistributionConfig] = None) -> np.ndarray:
        """Add time-dependent variation to feature values.

        Args:
            base_values: Base feature values for each sample
            n_timepoints: Number of timepoints
            dist_config: Distribution configuration for temporal effects
        """
        n_samples = len(base_values)
        time_points = np.arange(n_timepoints)

        # Generate temporal effects based on distribution if provided
        if dist_config and dist_config.name in self.CONTINUOUS_DISTRIBUTIONS:
            time_effect = self.CONTINUOUS_DISTRIBUTIONS[dist_config.name](
                self.rng,
                n_samples * n_timepoints,
                dist_config.params
            ).reshape(n_samples, n_timepoints)
            time_effect *= self.temporal_effect_size
        else:
            # Default linear time effect
            time_effect = self.temporal_effect_size * np.tile(time_points, (n_samples, 1))

        repeated_base = np.repeat(base_values, n_timepoints).reshape(n_samples, n_timepoints)
        return (repeated_base + time_effect).flatten()

    def generate_continuous(
            self,
            n_samples: int,
            n_features: int,
            prefix: str = 'cont',
            n_timepoints: int = 1,
            distributions: Optional[List[DistributionConfig]] = None,
            default_dist: Optional[DistributionConfig] = None,
            temporal_distributions: Optional[List[DistributionConfig]] = None
    ) -> Dict[str, np.ndarray]:
        """Generate continuous features with temporal variation and specified distributions.

        Args:
            n_samples: Number of unique samples
            n_features: Number of features to generate
            prefix: Prefix for feature names
            n_timepoints: Number of timepoints per sample
            distributions: List of distribution configurations for base features
            default_dist: Default distribution configuration if none specified
            temporal_distributions: List of distribution configurations for temporal effects
        """
        if distributions is None and default_dist is None:
            default_dist = DistributionConfig(name='normal', params={'loc': 0, 'scale': 1})

        features = {}
        for i in range(n_features):
            # Get base distribution configuration
            if distributions and i < len(distributions):
                dist_config = distributions[i]
                if isinstance(dist_config, str) and dist_config.lower() == 'random':
                    dist_config = self._get_random_continuous_dist()
            else:
                dist_config = default_dist

            if dist_config.name not in self.CONTINUOUS_DISTRIBUTIONS:
                raise ValueError(f"Unsupported distribution: {dist_config.name}")

            # Generate base values
            base_values = self.CONTINUOUS_DISTRIBUTIONS[dist_config.name](
                self.rng,
                n_samples,
                dist_config.params
            )

            # Get temporal distribution configuration
            temporal_dist = None
            if temporal_distributions and i < len(temporal_distributions):
                temp_dist = temporal_distributions[i]
                if isinstance(temp_dist, str) and temp_dist.lower() == 'random':
                    temporal_dist = self._get_random_continuous_dist()
                else:
                    temporal_dist = temp_dist

            # Apply temporal variation
            if n_timepoints > 1:
                values = self.add_temporal_variation(base_values, n_timepoints, temporal_dist)
            else:
                values = base_values

            features[f'{prefix}_{i}'] = values

        return features

    def generate_categorical(
            self,
            n_samples: int,
            n_features: int,
            min_cat: int = 2,
            max_cat: int = 5,
            prefix: str = 'cat',
            n_timepoints: int = 1,
            distributions: Optional[List[str]] = None,
            default_dist: str = 'uniform',
            transition_matrices: Optional[List[np.ndarray]] = None
    ) -> Dict[str, pd.Series]:
        """Generate categorical features with temporal changes and specified distributions.

        Args:
            n_samples: Number of unique samples
            n_features: Number of features to generate
            min_cat: Minimum number of categories
            max_cat: Maximum number of categories
            prefix: Prefix for feature names
            n_timepoints: Number of timepoints per sample
            distributions: List of distribution names for each feature
            default_dist: Default distribution if none specified
            transition_matrices: Optional list of transition probability matrices for temporal changes
        """
        features = {}
        total_samples = n_samples * n_timepoints

        for i in range(n_features):
            n_categories = self.rng.randint(min_cat, max_cat + 1)

            # Get distribution configuration
            if distributions and i < len(distributions):
                dist_name = distributions[i]
                if dist_name.lower() == 'random':
                    dist_name = self._get_random_categorical_dist()
            else:
                dist_name = default_dist

            if dist_name not in self.CATEGORICAL_DISTRIBUTIONS:
                raise ValueError(f"Unsupported distribution: {dist_name}")

            # Generate initial values
            if n_timepoints > 1:
                base_values = self.CATEGORICAL_DISTRIBUTIONS[dist_name](
                    self.rng,
                    n_samples,
                    n_categories
                )
                values = np.repeat(base_values, n_timepoints)

                # Apply temporal changes
                if transition_matrices and i < len(transition_matrices):
                    trans_matrix = transition_matrices[i]
                    for t in range(1, n_timepoints):
                        idx = t * n_samples
                        prev_values = values[idx - n_samples:idx]
                        for j in range(n_samples):
                            if self.rng.random() < self.categorical_change_prob:
                                values[idx + j] = self.rng.choice(
                                    range(n_categories),
                                    p=trans_matrix[prev_values[j]]
                                )
                else:
                    # Use default random transitions
                    change_mask = self.rng.random(total_samples) < self.categorical_change_prob
                    values[change_mask] = self.CATEGORICAL_DISTRIBUTIONS[dist_name](
                        self.rng,
                        sum(change_mask),
                        n_categories
                    )
            else:
                values = self.CATEGORICAL_DISTRIBUTIONS[dist_name](
                    self.rng,
                    total_samples,
                    n_categories
                )

            features[f'{prefix}_{i}'] = pd.Series(values).astype('category')

        return features


class EffectGenerator:
    """Generates and applies various effects to features with interaction scaling."""

    def __init__(self, rng: np.random.RandomState, effect_size: float = 1.0):
        """Initialize effect generator.

        Args:
            rng: Random number generator
            effect_size: Base effect size for feature coefficients
        """
        self.rng = rng
        self.effect_size = effect_size

    def generate_feature_effects(self, n_features: int) -> np.ndarray:
        """Generate main feature coefficients with proper scaling.

        Args:
            n_features: Number of features

        Returns:
            Array of feature coefficients
        """
        return self.rng.normal(0, self.effect_size, n_features)

    def generate_subgroup_effects(
            self,
            predictive_features: List[str],
            subgroups: List[Tuple[str, str]],
            subgroup_effect_size: float,
            feature_specific_subgroup_scale: float,
            hierarchical_effect_scale: float
    ) -> Dict[Tuple[str, str], float]:
        """Generate subgroup-specific effects.

        Args:
            subgroups: List of (sex, age_group) tuples defining unique subgroups
            subgroup_effect_size: Relative size of subgroup-specific effects
            feature_specific_subgroup_scale: Scaling factor for feature-specific subgroup effects
            hierarchical_effect_scale: Scaling factor for hierarchical effects

        Returns:
            Dictionary mapping feature names to subgroup-specific effects
        """
        effects = {
            'main': {},
            'feature_specific': {},
            'hierarchical': {},
            'interactions': {}
        }

        # Generate main subgroup effects
        for sex, age_group in itertools.product(['M', 'F'], subgroups['age_group']):
            effects['main'][(sex, age_group)] = self.rng.normal(
                0, subgroup_effect_size
            )

        # Generate feature-specific subgroup effects
        for feature in predictive_features:
            effects['feature_specific'][feature] = {}
            for sex, age_group in itertools.product(['M', 'F'], subgroups['age_group']):
                effects['feature_specific'][feature][(sex, age_group)] = self.rng.normal(
                    0, subgroup_effect_size * feature_specific_subgroup_scale
                )

        # Generate hierarchical effects
        for sex in ['M', 'F']:
            effects['hierarchical'][sex] = self.rng.normal(
                0, subgroup_effect_size * hierarchical_effect_scale
            )

        for age_group in subgroups['age_group']:
            effects['hierarchical'][age_group] = self.rng.normal(
                0, subgroup_effect_size * hierarchical_effect_scale
            )

        return effects

    @staticmethod
    def apply_subgroup_effects(
            logits: np.ndarray,
            subgroup_effects: Dict,
            features: Dict[str, np.ndarray],
            data: pd.DataFrame
    ) -> np.ndarray:
        """Apply enhanced subgroup effects to logits."""
        feature_logits = np.zeros_like(logits)

        # Apply main subgroup effects
        for (sex, age_group), effect in subgroup_effects['main'].items():
            mask = (data['sex'] == sex) & (data['age_group'] == age_group)
            feature_logits[mask] += effect

        # Apply feature-specific subgroup effects
        for feature, effects in subgroup_effects['feature_specific'].items():
            feature_values = features[feature]
            for (sex, age_group), effect in effects.items():
                mask = (data['sex'] == sex) & (data['age_group'] == age_group)
                feature_logits[mask] += effect * feature_values[mask]

        # Apply hierarchical effects
        for sex, effect in subgroup_effects['hierarchical'].items():
            if isinstance(sex, str) and sex in ['M', 'F']:
                mask = (data['sex'] == sex)
                feature_logits[mask] += effect

        for age_group, effect in subgroup_effects['hierarchical'].items():
            if isinstance(age_group, str) and age_group in data['age_group'].unique():
                mask = (data['age_group'] == age_group)
                feature_logits[mask] += effect

        return logits + feature_logits

    def generate_interactions(self, features: List[str],
                              max_order: int,
                              interaction_prob: float,
                              interaction_effect_size: float) -> List[Tuple[List[str], float]]:
        """Generate random interaction terms.

        Args:
            features: List of feature names
            max_order: Maximum number of features in an interaction
            interaction_prob: Probability of including each possible interaction
            interaction_effect_size: Size of interaction effects

        Returns:
            List of (interacting_features, coefficient) tuples
        """
        interactions = []

        # Generate all possible combinations up to max_order
        for order in range(2, max_order + 1):
            combinations = list(itertools.combinations(features, order))
            n_possible = len(combinations)

            # Scale interaction effects by order and number of possible interactions
            scaled_effect_size = (interaction_effect_size / (np.sqrt(order) * np.sqrt(n_possible)))

            for combo in combinations:
                if self.rng.random() < interaction_prob:
                    coef = self.rng.normal(0, scaled_effect_size)
                    interactions.append((list(combo), coef))

        return interactions

    @staticmethod
    def apply_nonlinear_effects(
            feature: np.ndarray,
            coef: float,
            transformations: List[str] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply nonlinear transformations to a feature.

        Args:
            feature: Feature values
            coef: Base coefficient for the feature
            transformations: List of transformation types to apply (default: quadratic and sine)

        Returns:
            Tuple of (transformed effects, transformation coefficients)
        """
        if transformations is None:
            transformations = ['quadratic', 'sine']

        n_transforms = len(transformations)
        effects = np.zeros_like(feature)
        transform_coefs = {}

        # Scale coefficients by number of transformations
        scale_factor = 1 / np.sqrt(n_transforms)

        for transform in transformations:
            if transform == 'quadratic':
                transform_coefs['quadratic'] = -0.5 * coef * scale_factor
                effects += transform_coefs['quadratic'] * feature ** 2
            elif transform == 'sine':
                transform_coefs['sine'] = 0.3 * coef * scale_factor
                effects += transform_coefs['sine'] * np.sin(feature)
            elif transform == 'exponential':
                transform_coefs['exponential'] = 0.1 * coef * scale_factor
                effects += transform_coefs['exponential'] * np.exp(-np.abs(feature))

        return effects, transform_coefs


class SyntheticDataGenerator:
    """Main class for generating synthetic clinical data."""

    def __init__(self, config: DataGenerationConfig):
        """Initialize the generator with configuration."""
        self.config = config
        self.rng = np.random.RandomState(config.random_state)

        # Initialize feature generators
        if config.n_timepoints > 1:
            self.feature_generator = TemporalFeatureGenerator(
                self.rng,
                config.temporal_effect_size,
                config.categorical_change_prob
            )
        else:
            self.feature_generator = FeatureGenerator(self.rng)

        # Initialize effect generator
        self.effect_generator = EffectGenerator(self.rng, config.feature_effect_size)

        # Initialize relationships dictionary
        self.true_relationships: Dict = {}

        # Initialize data storage
        self.data: Optional[pd.DataFrame] = None

    def generate(self) -> Tuple[pd.DataFrame, Dict]:
        """Generate synthetic clinical data according to configuration.

        Returns:
            Tuple containing:
            - DataFrame with synthetic data
            - Dictionary of true relationships used to generate the data
        """
        self.data = pd.DataFrame()

        # Generate data components
        self._generate_structure()
        self._generate_features()
        self._generate_outcome()

        # Add missing values if requested
        if self.config.missing_rate > 0:
            self._add_missing_values()

        return self.data, self.true_relationships

    def _generate_structure(self) -> None:
        """Generate basic data structure including IDs, site assignments, and demographics."""
        # Generate patient IDs and timepoints
        if self.config.n_timepoints > 1:
            self.data['patient_id'] = np.repeat(range(self.config.n_samples),
                                                self.config.n_timepoints)
            self.data['timepoint'] = np.tile(range(self.config.n_timepoints),
                                             self.config.n_samples)

        # Generate site assignments
        site_ids = self.rng.choice(range(self.config.n_sites),
                                   size=self.config.n_samples)
        if self.config.n_timepoints > 1:
            site_ids = np.repeat(site_ids, self.config.n_timepoints)
        sites = [f'site_{i}' for i in site_ids]
        sites = pd.Series(sites).astype('category')
        self.data['site'] = sites

        # Generate sex distribution
        sex = self.rng.choice(['M', 'F'],
                              size=self.config.n_samples,
                              p=[self.config.male_prob, 1 - self.config.male_prob])
        if self.config.n_timepoints > 1:
            sex = np.repeat(sex, self.config.n_timepoints)
        sex = pd.Series(sex).astype('category')
        self.data['sex'] = sex

        # Generate age distribution and groups
        age = self.rng.normal(self.config.age_ave,
                              self.config.age_std,
                              self.config.n_samples)
        age = np.clip(age, 0, 100)  # Ensure ages are between 0 and 100

        # Create age groups (10-year bins)
        age_groups = pd.cut(age,
                            bins=range(0, 101, 10),
                            labels=[f'{i}-{i + 9}' for i in range(0, 100, 10)],
                            right=False)

        if self.config.n_timepoints > 1:
            age_groups = np.repeat(age_groups, self.config.n_timepoints)
        age_groups = pd.Series(age_groups).astype('category')
        self.data['age_group'] = age_groups

    def _generate_features(self) -> None:
        """Generate all features according to configuration."""
        # Generate predictive features
        continuous_features = self.feature_generator.generate_continuous(
            self.config.n_samples,
            self.config.n_predictive_continuous,
            'pred_cont',
            self.config.n_timepoints,
            self.config.continuous_distributions,
            self.config.default_continuous_dist
        )

        categorical_features = self.feature_generator.generate_categorical(
            self.config.n_samples,
            self.config.n_predictive_categorical,
            self.config.cat_min,
            self.config.cat_max,
            'pred_cat',
            self.config.n_timepoints,
            self.config.categorical_distributions,
            self.config.default_categorical_dist
        )

        # Generate noise features
        noise_continuous = self.feature_generator.generate_continuous(
            self.config.n_samples,
            self.config.n_noise_continuous,
            'noise_cont',
            self.config.n_timepoints,
            self.config.continuous_distributions,
            self.config.default_continuous_dist
        )

        noise_categorical = self.feature_generator.generate_categorical(
            self.config.n_samples,
            self.config.n_noise_categorical,
            self.config.cat_min,
            self.config.cat_max,
            'noise_cat',
            self.config.n_timepoints,
            self.config.categorical_distributions,
            self.config.default_categorical_dist
        )

        # Add all features to DataFrame
        for name, values in {
            **continuous_features,
            **categorical_features,
            **noise_continuous,
            **noise_categorical
        }.items():
            self.data[name] = values

    def _generate_outcome(self) -> None:
        """Generate outcome variable based on features and effects."""
        # Initialize logits from feature effects only
        logits = np.zeros(len(self.data))

        # Add feature effects without site effects
        self._add_feature_effects(logits)

        # Generate site-specific thresholds to achieve target prevalences
        site_prevalences = np.linspace(
            self.config.site_prevalence_range[0],
            self.config.site_prevalence_range[1],
            self.config.n_sites
        )
        site_prevalences = self.rng.permutation(site_prevalences)

        # Store target prevalences
        self.true_relationships['site_effects'] = dict(zip(
            [f'site_{i}' for i in range(self.config.n_sites)],
            site_prevalences
        ))

        # Convert logits to probabilities
        base_probs = sigmoid(logits)

        # Initialize outcome array
        self.data['outcome'] = np.zeros(len(self.data))

        # Apply site-specific thresholding with noise
        for site in range(self.config.n_sites):
            site_mask = (self.data['site'] == f'site_{site}')
            site_data = base_probs[site_mask]

            if len(site_data) == 0:
                continue

            # Calculate target number of positive cases
            target_positives = int(len(site_data) * site_prevalences[site])

            # Add random noise to probabilities to avoid deterministic cutoffs
            noise = self.rng.normal(0, self.config.classification_noise, len(site_data))
            noisy_probs = np.clip(site_data + noise, 0, 1)

            # Sort probabilities to find threshold
            sorted_probs = np.sort(noisy_probs)
            if target_positives > 0:
                # Find probability threshold that gives desired prevalence
                threshold = sorted_probs[-target_positives]
            else:
                threshold = 1.0

            self.data.loc[site_mask, 'outcome'] = (noisy_probs >= threshold).astype(int)

    def _add_feature_effects(self, logits: np.ndarray) -> None:
        """Add feature effects to the logits."""
        # Get list of predictive features
        continuous_features = [f'pred_cont_{i}' for i in range(self.config.n_predictive_continuous)]
        categorical_features = [f'pred_cat_{i}' for i in range(self.config.n_predictive_categorical)]
        predictive_features = continuous_features + categorical_features

        # Initialize effects
        feature_logits = np.zeros_like(logits)

        # Get unique subgroups
        subgroups = {
            'age_group': self.data['age_group'].unique(),
            'sex': self.data['sex'].unique()
        }

        # Generate subgroup effects
        subgroup_effects = self.effect_generator.generate_subgroup_effects(
            predictive_features=predictive_features,
            subgroups=subgroups,
            subgroup_effect_size=self.config.subgroup_effect_size,
            feature_specific_subgroup_scale=self.config.feature_specific_subgroup_scale,
            hierarchical_effect_scale=self.config.hierarchical_effect_scale
        )

        # Generate main effects with base scaling
        n_total_features = len(predictive_features)
        base_scale = 1.0 / np.sqrt(max(1, n_total_features))

        # Generate main effects
        continuous_coefs = self.effect_generator.generate_feature_effects(
            self.config.n_predictive_continuous
        ) * base_scale
        categorical_coefs = self.effect_generator.generate_feature_effects(
            self.config.n_predictive_categorical
        ) * base_scale

        # Create features dictionary for subgroup effects
        features_dict = {}
        for i, feature in enumerate(continuous_features):
            features_dict[feature] = self.data[feature].values
        for i, feature in enumerate(categorical_features):
            features_dict[feature] = self.data[feature].cat.codes.values

        # Apply subgroup effects
        feature_logits = self.effect_generator.apply_subgroup_effects(
            feature_logits, subgroup_effects, features_dict, self.data
        )

        # Add continuous feature effects
        nonlinear_transformations = {}
        for i, coef in enumerate(continuous_coefs):
            feature = self.data[f'pred_cont_{i}']
            feature_logits += coef * feature

            if self.config.include_nonlinear:
                nonlinear_effects, transformations = self.effect_generator.apply_nonlinear_effects(
                    feature, coef
                )
                feature_logits += nonlinear_effects
                nonlinear_transformations[f'pred_cont_{i}'] = transformations

        # Add categorical feature effects
        for i, coef in enumerate(categorical_coefs):
            feature = self.data[f'pred_cat_{i}'].cat.codes.values
            feature_logits += coef * feature

        if self.config.include_interactions:
            interactive_features = predictive_features + ['age_group', 'sex']
            if self.config.site_specific_effects:
                interactive_features += ['site']
            interactions = self.effect_generator.generate_interactions(
                interactive_features,
                self.config.max_interaction_order,
                self.config.interaction_probability,
                self.config.interaction_effect_size
            )
        else:
            interactions = []

        # Add interaction effects
        for features, coef in interactions:
            interaction_term = self._calculate_interaction_term(features)
            feature_logits += coef * interaction_term

        logits += feature_logits

        # Store relationships
        self._store_relationships(
            continuous_features, continuous_coefs,
            categorical_features, categorical_coefs,
            subgroup_effects, interactions,
            nonlinear_transformations if self.config.include_nonlinear else None
        )

    def _calculate_interaction_term(self, features: List[str]) -> np.ndarray:
        """Calculate interaction term for given features."""
        feature = features[0]
        if feature in ['age_group', 'sex', 'site'] or feature.startswith('pred_cat'):
            interaction_term = self.data[features[0]].cat.codes.values
        else:
            interaction_term = self.data[features[0]].values

        for feature in features[1:]:
            if feature in ['age_group', 'sex', 'site'] or feature.startswith('pred_cat'):
                interaction_term = interaction_term * self.data[feature].cat.codes.values
            else:
                interaction_term = interaction_term * self.data[feature].values

        return interaction_term

    def _store_relationships(self, continuous_features, continuous_coefs,
                             categorical_features, categorical_coefs,
                             subgroup_effects, interactions,
                             nonlinear_transformations=None):
        """Store all relationships in the true_relationships dictionary."""
        self.true_relationships.update({
            'continuous_coefficients': dict(zip(continuous_features, continuous_coefs)),
            'categorical_coefficients': dict(zip(categorical_features, categorical_coefs)),
            'subgroup_effects': subgroup_effects,
            'interactions': [{'features': features, 'coefficient': coef}
                             for features, coef in interactions],
            'noise_features': {
                'continuous': [f'noise_cont_{i}'
                               for i in range(self.config.n_noise_continuous)],
                'categorical': [f'noise_cat_{i}'
                                for i in range(self.config.n_noise_categorical)]
            }
        })

        if nonlinear_transformations:
            self.true_relationships['nonlinear_effects'] = nonlinear_transformations

    def _add_missing_values(self) -> None:
        """Add missing values to features."""
        feature_cols = [col for col in self.data.columns
                        if col not in ['outcome', 'site', 'patient_id',
                                       'timepoint', 'sex', 'age_group']]

        for col in feature_cols:
            missing_mask = self.rng.random(len(self.data)) < self.config.missing_rate
            self.data.loc[missing_mask, col] = np.nan

    def get_feature_importance(self) -> pd.DataFrame:
        """Calculate feature importance based on coefficients."""
        if not self.true_relationships:
            raise RuntimeError("No relationships available. Generate data first.")

        importances = []

        # Add main feature importances
        for feat_type in ['continuous', 'categorical']:
            coef_dict = self.true_relationships[f'{feat_type}_coefficients']
            for feat, coef in coef_dict.items():
                importance = abs(coef)
                if (self.config.include_nonlinear and
                        feat_type == 'continuous' and
                        feat in self.true_relationships.get('nonlinear_effects', {})):
                    nonlinear_coefs = self.true_relationships['nonlinear_effects'][feat]
                    importance += sum(abs(v) for v in nonlinear_coefs.values())
                importances.append({
                    'feature': feat,
                    'importance': importance,
                    'type': feat_type
                })

        # Add interaction importances
        for interaction in self.true_relationships['interactions']:
            importances.append({
                'feature': '+'.join(interaction['features']),
                'importance': abs(interaction['coefficient']),
                'type': 'interaction'
            })

        # Add subgroup importances from the new structure
        subgroup_effects = self.true_relationships.get('subgroup_effects', {})

        # Main subgroup effects
        for (sex, age_group), coef in subgroup_effects.get('main', {}).items():
            importances.append({
                'feature': f'subgroup_main_{sex}_{age_group}',
                'importance': abs(coef),
                'type': 'subgroup_main'
            })

        # Feature-specific subgroup effects
        for feature, effects in subgroup_effects.get('feature_specific', {}).items():
            for (sex, age_group), coef in effects.items():
                importances.append({
                    'feature': f'subgroup_feature_{feature}_{sex}_{age_group}',
                    'importance': abs(coef),
                    'type': 'subgroup_feature'
                })

        # Hierarchical effects
        for group, coef in subgroup_effects.get('hierarchical', {}).items():
            importances.append({
                'feature': f'hierarchical_{group}',
                'importance': abs(coef),
                'type': 'subgroup_hierarchical'
            })

        return pd.DataFrame(importances).sort_values('importance', ascending=False)

    def get_subgroup_statistics(self) -> pd.DataFrame:
        """Calculate statistics for each subgroup."""
        if self.data is None:
            raise RuntimeError("No data available. Generate data first.")

        stats = []
        for sex in ['M', 'F']:
            for age_group in self.data['age_group'].unique():
                mask = (self.data['sex'] == sex) & (self.data['age_group'] == age_group)
                subgroup_data = self.data[mask]

                stats.append({
                    'sex': sex,
                    'age_group': age_group,
                    'n_samples': len(subgroup_data),
                    'n_patients': (subgroup_data['patient_id'].nunique()
                                   if 'patient_id' in self.data else len(subgroup_data)),
                    'outcome_rate': subgroup_data['outcome'].mean(),
                    'missing_rate': subgroup_data.iloc[:, 3:].isnull().mean().mean()
                })

        return pd.DataFrame(stats).sort_values(['sex', 'age_group']).reset_index(drop=True)

    def get_site_statistics(self) -> pd.DataFrame:
        """Calculate site-specific statistics."""
        if self.data is None:
            raise RuntimeError("No data available. Generate data first.")

        stats = []
        for site in sorted(self.data['site'].unique()):
            site_mask = self.data['site'] == site
            site_data = self.data[site_mask]

            stat = {
                'site': site,
                'n_samples': len(site_data),
                'outcome_rate': site_data['outcome'].mean(),
                'missing_rate': site_data.iloc[:, 3:].isnull().mean().mean()
            }

            stats.append(stat)

        return pd.DataFrame(stats)


def main():
    # Create configuration
    config = DataGenerationConfig(
        n_samples=1000,
        n_timepoints=3,
        n_predictive_continuous=3,
        n_predictive_categorical=2,
        include_interactions=True,
        include_nonlinear=True,
        missing_rate=0.1,
        random_state=42
    )

    # Initialize and run generator
    generator = SyntheticDataGenerator(config)
    data, relationships = generator.generate()

    # Print summary statistics
    print("\nData Shape:", data.shape)
    print("\nFeature Importance:")
    print(generator.get_feature_importance())
    print("\nSite Statistics:")
    print(generator.get_site_statistics())
    print("\nSubgroup Statistics:")
    print(generator.get_subgroup_statistics())

    if config.n_timepoints > 1:
        print("\nTemporal Trends:")
        print(data.groupby('timepoint')['outcome'].mean())


if __name__ == "__main__":
    main()
