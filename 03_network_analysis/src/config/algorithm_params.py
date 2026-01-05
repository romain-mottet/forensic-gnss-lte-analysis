from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class AlgorithmParams:
    
    context: str
    
    # ========================================================================
    # PHASE 2.2: Distance Estimation Parameters
    # ========================================================================
    
    # Distance calculation method: "path_loss" or "formula"
    distance_calculation_method: str
    
    # Formula-based distance estimation (used when method = "formula")
    formula_coefficients: Optional[Dict] = None
    
    # Path loss exponent per frequency band (used when method = "path_loss")
    path_loss_exp_by_freq: dict = None
    tx_power_by_freq: dict = None
    freq_band_map: dict = None
    earfcn_band_map: dict = None
    
    # Confidence and uncertainty thresholds
    rsrp_thresholds: dict = None  # Maps RSRP ranges to confidence levels
    max_uncertainty_by_level: dict = None  # Max uncertainty per confidence level
    min_uncertainty_m: float = 30.0  # Minimum uncertainty in meters
    
    # ========================================================================
    # PHASE 2.3: Bearing Estimation Parameters
    # ========================================================================
    
    # Multi-factor confidence scoring
    rsrq_score_thresholds: dict = None  # RSRQ value -> points
    rssi_boost_thresholds: dict = None  # RSSI thresholds and bonus points
    accuracy_boost_thresholds: dict = None  # GPS accuracy thresholds and bonus
    ta_boost_thresholds: dict = None  # TA (timing advance) thresholds
    speed_boost_thresholds: dict = None  # Speed thresholds and bonus
    
    # Confidence level mapping
    bearing_confidence_thresholds: dict = None  # Score ranges -> (confidence, uncertainty)
    
    # ========================================================================
    # PHASE 3.1: Trilateration Input Preparation
    # ========================================================================
    
    rsrp_quality_threshold_dbm: float = -120.0  # RSRP threshold for filtering
    min_cells_required: int = 3  # Minimum cells for trilateration
    max_cells_to_keep: int = 5  # Maximum cells per timestamp
    
    # ========================================================================
    # PHASE 3.2: Trilateration Solver Parameters
    # ========================================================================
    
    # Weighting and convergence
    weight_by_uncertainty_power: float = 2.0  # Weight = 1 / uncertainty^power
    convergence_threshold_m: float = 1.0  # Convergence threshold in meters
    max_iterations: int = 100  # Maximum solver iterations
    
    # GDOP and residual thresholds
    max_gdop_accepted: float = 10.0  # Maximum acceptable GDOP
    residual_weight: float = 0.1  # Weight for residual in solution quality
    
    # ========================================================================
    # PHASE 3.3: Ground Truth Validation Parameters
    # ========================================================================
    
    # Accuracy metrics
    cep_percentile: float = 50.0  # CEP = circular error probability (50th percentile)
    r95_percentile: float = 95.0  # R95 = 95th percentile
    
    @classmethod
    def create_default(cls, context: str = "default") -> "AlgorithmParams":
        """Create default parameters for a given context."""
        raise NotImplementedError("Subclasses must implement create_default()")


class DefaultParams(AlgorithmParams):
    """Default parameters (backward compatible with existing code)."""
    
    @classmethod
    def create_default(cls, context: str = "default") -> "DefaultParams":
        return DefaultParams(
            context="default",
            distance_calculation_method="path_loss",
            
            # ====== PHASE 2.2: Distance Estimation ======
            path_loss_exp_by_freq={
                (600, 800): 2.75,    # 700 MHz
                (750, 850): 2.75,    # 800 MHz
                (850, 950): 2.72,    # 900 MHz
                (1700, 1900): 2.78,  # 1800 MHz
                (2000, 2200): 2.80,  # 2100 MHz
                (2500, 2700): 2.82,  # 2600 MHz
            },
            tx_power_by_freq={
                (600, 850): (20.0, "700/800 MHz LTE"),
                (850, 1000): (20.0, "900 MHz GSM"),
                (1700, 2000): (20.0, "1800 MHz DCS"),
                (2000, 2200): (20.0, "2100 MHz UMTS"),
                (2500, 2700): (18.0, "2600 MHz LTE (lower power)"),
            },
            freq_band_map={
                (600, 850): (28, 29.0),    # 700/800 MHz
                (850, 1000): (8, 31.5),    # 900 MHz
                (1700, 2000): (3, 37.5),   # 1800 MHz
                (2000, 2200): (1, 38.9),   # 2100 MHz
                (2500, 2700): (7, 40.7),   # 2600 MHz
            },
            earfcn_band_map={
                "band_1": (0, 599, 2100.0, 38.9),      # 2100 MHz
                "band_3": (600, 1199, 1800.0, 37.5),   # 1800 MHz
                "band_7": (1200, 1949, 2600.0, 40.7),  # 2600 MHz
                "band_8": (3400, 3799, 900.0, 31.5),   # 900 MHz
                "band_20": (6000, 6149, 800.0, 30.5),  # 800 MHz
                "band_28": (9000, 9599, 700.0, 29.0),  # 700 MHz
            },
            rsrp_thresholds={
                (-85, float('inf')): ("EXCELLENT", 0.08),
                (-90, -85): ("HIGH", 0.10),
                (-100, -90): ("GOOD", 0.15),
                (-110, -100): ("MEDIUM", 0.20),
                (-120, -110): ("WEAK", 0.25),
                (float('-inf'), -120): ("VERY_WEAK", 0.30),
            },
            max_uncertainty_by_level={
                "EXCELLENT": 100.0,
                "HIGH": 120.0,
                "GOOD": 150.0,
                "MEDIUM": 200.0,
                "WEAK": 300.0,
                "VERY_WEAK": 400.0,
            },
            min_uncertainty_m=30.0,
            
            # ====== PHASE 2.3: Bearing Estimation ======
            rsrq_score_thresholds={
                10: 50,   # RSRQ >= 10 dB
                5: 30,    # RSRQ >= 5 dB
                -5: 20,   # RSRQ >= -5 dB
            },
            rssi_boost_thresholds={
                -70: 15,  # RSSI > -70 dBm
                -80: 7,   # RSSI > -80 dBm
            },
            accuracy_boost_thresholds={
                5: 15,    # Accuracy <= 5 m
                10: 10,   # Accuracy <= 10 m
                20: 5,    # Accuracy <= 20 m
            },
            ta_boost_thresholds={
                2: 10,    # TA <= 2 slots
                5: 5,     # TA <= 5 slots
            },
            speed_boost_thresholds={
                5: 10,    # Speed <= 5 km/h
                15: 5,    # Speed <= 15 km/h
            },
            bearing_confidence_thresholds={
                (60, float('inf')): ("HIGH", 30),
                (40, 60): ("MEDIUM", 60),
                (float('-inf'), 40): ("LOW", 90),
            },
            
            # ====== PHASE 3.1: Trilateration Input ======
            rsrp_quality_threshold_dbm=-120.0,
            min_cells_required=3,
            max_cells_to_keep=5,
            
            # ====== PHASE 3.2: Trilateration Solver ======
            weight_by_uncertainty_power=2.0,
            convergence_threshold_m=1.0,
            max_iterations=100,
            max_gdop_accepted=10.0,
            residual_weight=0.1,
            
            # ====== PHASE 3.3: Ground Truth Validation ======
            cep_percentile=50.0,
            r95_percentile=95.0,
        )


class CityParams(AlgorithmParams):
    """Parameters optimized for urban environments (high signal density)."""
    
    @classmethod
    def create_default(cls, context: str = "city") -> "CityParams":
        # Start with default and adjust for urban environment
        params = DefaultParams.create_default()
        
        # In cities: tighter thresholds, smaller uncertainty
        params.context = "city"
        params.rsrp_quality_threshold_dbm = -110.0  # Higher threshold (stricter)
        params.min_cells_required = 4  # More cells available
        params.max_cells_to_keep = 8  # Use more cells for better accuracy
        
        # Distance estimation: tighter uncertainty bounds
        params.max_uncertainty_by_level = {
            "EXCELLENT": 80.0,   # Reduced from 100
            "HIGH": 100.0,       # Reduced from 120
            "GOOD": 120.0,       # Reduced from 150
            "MEDIUM": 150.0,     # Reduced from 200
            "WEAK": 250.0,       # Reduced from 300
            "VERY_WEAK": 350.0,  # Reduced from 400
        }
        
        # Bearing: tighter uncertainty for urban
        params.bearing_confidence_thresholds = {
            (60, float('inf')): ("HIGH", 25),
            (40, 60): ("MEDIUM", 50),
            (float('-inf'), 40): ("LOW", 75),
        }
        
        # Solver: stricter convergence, lower GDOP threshold
        params.convergence_threshold_m = 0.5
        params.max_gdop_accepted = 8.0
        
        return params


class VillageParams(AlgorithmParams):
    """Parameters optimized for rural environments (moderate signal density)."""
    
    @classmethod
    def create_default(cls, context: str = "village") -> "VillageParams":
        # Start with default
        params = DefaultParams.create_default()
        
        # In villages: balanced thresholds
        params.context = "village"
        params.rsrp_quality_threshold_dbm = -120.0  # Standard threshold
        params.min_cells_required = 2  # Standard minimum
        params.max_cells_to_keep = 5  # Standard maximum
        
        # Distance estimation: balanced uncertainty
        params.max_uncertainty_by_level = {
            "EXCELLENT": 100.0,
            "HIGH": 120.0,
            "GOOD": 150.0,
            "MEDIUM": 200.0,
            "WEAK": 300.0,
            "VERY_WEAK": 400.0,
        }
        
        # Bearing: standard uncertainty
        params.bearing_confidence_thresholds = {
            (60, float('inf')): ("HIGH", 30),
            (40, 60): ("MEDIUM", 60),
            (float('-inf'), 40): ("LOW", 90),
        }
        
        # Solver: standard parameters
        params.convergence_threshold_m = 1.0
        params.max_gdop_accepted = 10.0
        
        return params


class TownParams(AlgorithmParams):
    """Parameters optimized for semi-urban environments (moderate-high signal density)."""
    
    @classmethod
    def create_default(cls, context: str = "town") -> "TownParams":
        # Start with default
        params = DefaultParams.create_default()
        
        # In towns: moderately strict thresholds
        params.context = "town"
        params.rsrp_quality_threshold_dbm = -115.0  # Moderately strict
        params.min_cells_required = 3  # Standard minimum
        params.max_cells_to_keep = 6  # Slightly more than default
        
        # Distance estimation: slightly tighter uncertainty
        params.max_uncertainty_by_level = {
            "EXCELLENT": 90.0,
            "HIGH": 110.0,
            "GOOD": 140.0,
            "MEDIUM": 180.0,
            "WEAK": 280.0,
            "VERY_WEAK": 380.0,
        }
        
        # Bearing: slightly tighter uncertainty
        params.bearing_confidence_thresholds = {
            (60, float('inf')): ("HIGH", 28),
            (40, 60): ("MEDIUM", 55),
            (float('-inf'), 40): ("LOW", 85),
        }
        
        # Solver: moderately strict
        params.convergence_threshold_m = 0.75
        params.max_gdop_accepted = 9.0
        
        return params


class FormulaParams(AlgorithmParams):
    """Parameters for formula-based distance estimation (regression models)."""
    
    @classmethod
    def create_default(cls, context: str = "formula", subcontext: Optional[str] = None, formula_data: Optional[Dict] = None) -> "FormulaParams":

        # Get base params from subcontext for bearing/trilateration
        subcontext = (subcontext or "default").lower().strip()
        
        if subcontext == "city":
            base_params = CityParams.create_default("city")
        elif subcontext == "village":
            base_params = VillageParams.create_default("village")
        elif subcontext == "town":
            base_params = TownParams.create_default("town")
        else:
            base_params = DefaultParams.create_default("default")
        
        # Override distance calculation method to formula
        base_params.context = f"formula_{subcontext}"
        base_params.distance_calculation_method = "formula"
        base_params.formula_coefficients = formula_data
        
        # Clear path_loss parameters (not used with formula method)
        base_params.path_loss_exp_by_freq = None
        base_params.tx_power_by_freq = None
        base_params.freq_band_map = None
        base_params.earfcn_band_map = None
        
        return base_params


def get_algorithm_params(context: Optional[str] = None, subcontext: Optional[str] = None, formula_data: Optional[Dict] = None) -> AlgorithmParams:
    """
    Factory function to retrieve algorithm parameters for a given context.
    
    Args:
        context: Environment context ("city", "village", "town", "formula", or None for default)
        subcontext: Subcontext for formula mode (bearing/trilateration context)
        formula_data: Dictionary with formula coefficients (required if context="formula")
    
    Returns:
        AlgorithmParams object with context-specific parameters
    
    Example:
        >>> params = get_algorithm_params("city")
        >>> print(params.max_cells_to_keep)
        8
        
        >>> params = get_algorithm_params()  # Returns default
        >>> print(params.context)
        "default"
        
        >>> formula_dict = {"intercept": 2.5, "coef_rsrp": -0.002, ...}
        >>> params = get_algorithm_params("formula", subcontext="town", formula_data=formula_dict)
        >>> print(params.distance_calculation_method)
        "formula"
        >>> print(params.context)
        "formula_town"
    """
    context = (context or "default").lower().strip()
    
    if context == "city":
        return CityParams.create_default("city")
    elif context == "village":
        return VillageParams.create_default("village")
    elif context == "town":
        return TownParams.create_default("town")
    elif context == "formula":
        return FormulaParams.create_default("formula", subcontext=subcontext, formula_data=formula_data)
    else:
        return DefaultParams.create_default("default")
