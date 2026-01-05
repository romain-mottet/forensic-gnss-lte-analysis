import re
from pathlib import Path
from typing import Dict, Optional, List
from collections import defaultdict


class Config:
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_LOGS_DIR = DATA_DIR / 'raw_logs'
    TOWER_DATA_DIR = DATA_DIR / 'tower_data'

    RESULT_DIR = PROJECT_ROOT / 'result'
    REPORTS_DIR = PROJECT_ROOT / 'reports'
    
    @staticmethod
    def extract_prefix(logname: str) -> str:
        basename = logname.replace('.txt', '')
        
        # Pattern: Extract letters before first underscore
        match = re.match(r'^([a-zA-Z]+)_', basename)
        if match:
            return match.group(1)
        
        # Fallback: if no underscore, return entire basename
        match = re.match(r'^([a-zA-Z]+)', basename)
        if match:
            return match.group(1)
        
        return 'others'
    
    @classmethod
    def setup_for_log(cls, logname: str, context: Optional[str] = None, subcontext: Optional[str] = None) -> Dict[str, Optional[Path]]:
        """Setup configuration for a specific log file with context support.
        
        Creates directory structure:
        - result/prefix_data/basename_data/context/ (ROOT LEVEL with context subfolder)
        - result/prefix_data/basename_data/formula/subcontext/ (for formula mode)
        - reports/prefix_reports/basename_reports/context/
        - reports/prefix_reports/basename_reports/formula/subcontext/ (for formula mode)
        - All files get _{basename}_{context} or _{basename}_formula_{subcontext} suffix
        
        Examples:
            python main.py ixelle_4.txt city
            -> result/ixelle_data/ixelle_4_data/city/
            -> parsed_ixelle_4_city.csv
            -> signal_data_ixelle_4_city.csv
            
            python main.py lln_5.txt formula town
            -> result/lln_data/lln_5_data/formula/town/
            -> parsed_lln_5_formula_town.csv
            -> signal_data_lln_5_formula_town.csv
        """
        from config.algorithm_params import get_algorithm_params
        
        basename = logname.replace('.txt', '')
        prefix = cls.extract_prefix(basename)
        resolved_context = (context or 'default').lower().strip()
        
        # Get algorithm parameters for this context
        if resolved_context == "formula" and subcontext:
            algorithm_params = get_algorithm_params(resolved_context, subcontext=subcontext)
            resolved_subcontext = (subcontext or 'default').lower().strip()
        else:
            algorithm_params = get_algorithm_params(resolved_context)
            resolved_subcontext = None
        
        # Create base directories
        cls.RAW_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TOWER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Data output directory: result/prefix_data/basename_data/
        prefix_result_dir = cls.RESULT_DIR / f"{prefix}_data"
        prefix_result_dir.mkdir(parents=True, exist_ok=True)
        
        basename_data_dir = prefix_result_dir / f"{basename}_data"
        basename_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Context subdirectory for data
        if resolved_context == "formula" and resolved_subcontext:
            # Nested structure: formula/town/
            formula_dir = basename_data_dir / "formula"
            formula_dir.mkdir(parents=True, exist_ok=True)
            log_data_dir = formula_dir / resolved_subcontext
            log_data_dir.mkdir(parents=True, exist_ok=True)
            file_suffix = f"_{basename}_formula_{resolved_subcontext}"
        else:
            # Flat structure: city/ or town/ or default/
            log_data_dir = basename_data_dir / resolved_context
            log_data_dir.mkdir(parents=True, exist_ok=True)
            file_suffix = f"_{basename}_{resolved_context}"
        
        # Reports directory: reports/prefix_reports/basename_reports/
        prefix_report_dir = cls.REPORTS_DIR / f"{prefix}_reports"
        prefix_report_dir.mkdir(parents=True, exist_ok=True)
        
        basename_report_dir = prefix_report_dir / f"{basename}_reports"
        basename_report_dir.mkdir(parents=True, exist_ok=True)
        
        # Context subdirectory for reports
        if resolved_context == "formula" and resolved_subcontext:
            # Nested structure: formula/town/
            formula_report_dir = basename_report_dir / "formula"
            formula_report_dir.mkdir(parents=True, exist_ok=True)
            report_dir = formula_report_dir / resolved_subcontext
            report_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Flat structure: city/ or town/ or default/
            report_dir = basename_report_dir / resolved_context
            report_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {
            # Input
            'input_log': cls.RAW_LOGS_DIR / logname,
            
            # Log-specific data directory (ROOT LEVEL with context subfolder)
            'log_data_dir': log_data_dir,
            
            # Phase 1 outputs (fixed: no duplicate basename)
            'parsed_csv': log_data_dir / f"parsed{file_suffix}.csv",
            
            # Phase 2.1 outputs
            'signal_data_csv': log_data_dir / f"signal_data{file_suffix}.csv",
            'signal_metadata_csv': log_data_dir / f"signal_metadata{file_suffix}.csv",
            'neighbor_signals_csv': log_data_dir / f"neighbor_signals{file_suffix}.csv",
            
            # Phase 2.2 output
            'distance_estimates_csv': log_data_dir / f"distance_estimates{file_suffix}.csv",
            
            # Phase 2.3 output
            'bearing_estimates_csv': log_data_dir / f"bearing_estimates{file_suffix}.csv",
            
            # Phase 3.1 output
            'trilateration_input_csv': log_data_dir / f"trilateration_input{file_suffix}.csv",
            
            # Phase 3.2 output
            'trilateration_results_csv': log_data_dir / f"trilateration_results{file_suffix}.csv",
            
            # Phase 3.3 outputs
            'validation_results_csv': log_data_dir / f"validation_results{file_suffix}.csv",
            'ground_truth_csv': log_data_dir / f"ground_truth{file_suffix}.csv",
            'analysis_metadata_json': log_data_dir / f"analysis_metadata{file_suffix}.json",
            
            # Reports (WITH CONTEXT SUBFOLDER)
            'accuracy_summary_txt': report_dir / f"accuracy_summary{file_suffix}.txt",
            'coverage_report': report_dir / f"coverage_report{file_suffix}.txt",
            'towers_status': report_dir / f"towers_status{file_suffix}.json",
            
            # Database files (shared - NO SUFFIX, NO CONTEXT SUBFOLDER)
            'towers_db': cls.TOWER_DATA_DIR / 'towers.json',
            'pci_db': cls.TOWER_DATA_DIR / 'pci.json',
            
            # Metadata
            'basename': basename,
            'prefix': prefix,
            'context': resolved_context,
            'subcontext': resolved_subcontext,
            'file_suffix': file_suffix,
            'report_dir': report_dir,
            'algorithm_params': algorithm_params,
        }
        
        return paths
    
    @classmethod
    def list_logs_by_prefix(cls) -> Dict[str, List[str]]:
        """List all log files organized by prefix."""
        if not cls.RAW_LOGS_DIR.exists():
            return {}
        
        prefix_dict = defaultdict(list)
        for logfile in sorted(cls.RAW_LOGS_DIR.glob('*.txt')):
            prefix = cls.extract_prefix(logfile.name)
            prefix_dict[prefix].append(logfile.name)
        
        return dict(prefix_dict)
    
    @staticmethod
    def load_formula_coefficients(prefix: str) -> dict:
        """
        Load formula coefficients from JSON file based on log file prefix.
        
        Args:
            prefix: Log file prefix (e.g., 'lln', 'waha', 'ixelle')
        
        Returns:
            Dictionary containing formula coefficients and metadata
        
        Raises:
            FileNotFoundError: If formula JSON file doesn't exist for the prefix
            ValueError: If JSON file is malformed
        
        Example:
            >>> coeffs = Config.load_formula_coefficients('lln')
            >>> print(coeffs['coefficients']['intercept'])
            2.668...
        """
        import json
        from pathlib import Path
        
        # Formula JSON location
        formulas_dir = Path(__file__).parent.parent.parent / "data" / "formulas"
        formula_file = formulas_dir / f"{prefix}_formula.json"
        
        if not formula_file.exists():
            available_formulas = [f.stem.replace('_formula', '') 
                                 for f in formulas_dir.glob('*_formula.json')]
            raise FileNotFoundError(
                f"❌ ERROR: Formula file not found for prefix '{prefix}'.\n"
                f"   Expected location: {formula_file}\n"
                f"   Available formula prefixes: {', '.join(available_formulas)}\n"
                f"   Please generate the formula for '{prefix}' first using the formula_finder project."
            )
        
        try:
            with open(formula_file, 'r') as f:
                formula_data = json.load(f)
            
            # Validate required fields
            required_fields = ['context', 'model', 'features', 'coefficients']
            missing_fields = [field for field in required_fields if field not in formula_data]
            if missing_fields:
                raise ValueError(
                    f"❌ ERROR: Formula file {formula_file} is missing required fields: {missing_fields}"
                )
            
            return formula_data
        
        except json.JSONDecodeError as e:
            raise ValueError(
                f"❌ ERROR: Formula file {formula_file} contains invalid JSON: {e}"
            )
