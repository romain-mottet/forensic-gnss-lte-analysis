import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


def setup_logging(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_towers(towers_file: str) -> Dict[str, Any]:
    path = Path(towers_file)
    
    if not path.exists():
        raise FileNotFoundError(f"Towers file not found: {towers_file}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {towers_file}: {e}")


def load_existing_pci(pci_file: str) -> Dict[str, Any]:
    path = Path(pci_file)
    
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {pci_file}: {e}")


def extract_pci_data(towers: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    pci_data = {}
    
    # Handle different tower data structures
    towers_list = []
    
    if isinstance(towers, dict):
        # Check for standard nested structures first
        if 'towers' in towers:
            towers_list = towers['towers'] if isinstance(towers['towers'], list) else [towers['towers']]
        elif 'data' in towers:
            towers_list = towers['data'] if isinstance(towers['data'], list) else [towers['data']]
        else:
            # Handle dictionary with tower IDs as keys
            is_tower_dict = True
            for value in towers.values():
                if isinstance(value, dict) and ('pci' in value or 'latitude' in value or 'longitude' in value):
                    towers_list.append(value)
                elif isinstance(value, (list, str, int, float)):
                    is_tower_dict = False
                    break
            
            if not is_tower_dict:
                towers_list = [towers]
    elif isinstance(towers, list):
        towers_list = towers
    
    for tower in towers_list:
        if not isinstance(tower, dict):
            continue
        
        # Extract PCI - try multiple possible field names
        pci = None
        for pci_field in ['pci', 'PCI', 'cell_id', 'cellid', 'CellID']:
            if pci_field in tower:
                pci = tower[pci_field]
                break
        
        if pci is None:
            continue
        
        # Convert PCI to string for consistent key handling
        pci_key = str(pci)
        
        # Extract coordinates (REQUIRED)
        latitude = tower.get('latitude') or tower.get('lat') or tower.get('Latitude')
        longitude = tower.get('longitude') or tower.get('lon') or tower.get('Longitude')
        
        if latitude is None or longitude is None:
            continue  # Skip if missing coordinates
        
        # Build PCI record with ONLY the 6 essential fields
        pci_record = {
            'pci': pci_key,
            'latitude': latitude,
            'longitude': longitude
        }
        
        # Add optional essential fields if they exist
        # band
        band = tower.get('band') or tower.get('Band')
        if band is not None:
            pci_record['band'] = band
        
        # operator (from mnc or operator field)
        operator = tower.get('operator') or tower.get('Operator') or tower.get('mnc')
        if operator is not None:
            pci_record['operator'] = operator
        
        # frequency_mhz (from downlink_frequency or frequency_band)
        frequency = tower.get('downlink_frequency') or tower.get('frequency_band') or tower.get('frequency_mhz')
        if frequency is not None:
            pci_record['frequency_mhz'] = frequency
        
        pci_data[pci_key] = pci_record
    
    return pci_data


def merge_pci_data(existing: Dict[str, Any], new_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], int, int]:
    new_records = 0
    updated_records = 0
    
    for pci_key, pci_record in new_data.items():
        if pci_key in existing:
            existing_record = existing[pci_key]
            
            for key, value in pci_record.items():
                if key not in existing_record or existing_record[key] is None:
                    existing_record[key] = value
            
            updated_records += 1
            existing[pci_key] = pci_record
            new_records += 1
    
    return existing, new_records, updated_records


def save_pci_database(pci_data: Dict[str, Any], output_file: str) -> bool:
    path = Path(output_file)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(pci_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        raise IOError(f"Failed to write PCI database: {e}")


def populate_pci_database(
    towers_file: str,
    output_file: str,
    verbose: bool = False
) -> Dict[str, Any]:

    logger = setup_logging(verbose)
    
    try:
        logger.debug(f"Loading towers from: {towers_file}")
        towers = load_towers(towers_file)
        
        logger.debug(f"Loading existing PCI data from: {output_file}")
        existing_pci = load_existing_pci(output_file)
        
        if existing_pci:
            logger.info(f"Found {len(existing_pci)} existing PCI records")
        
        logger.debug("Extracting PCI data from towers")
        new_pci_data = extract_pci_data(towers)
        
        if not new_pci_data:
            logger.warning("No PCI data found in towers file")
            return {
                'success': False,
                'error': 'No PCI data found in towers file',
                'new_records': 0,
                'updated_records': 0,
                'total_records': len(existing_pci)
            }
        
        logger.debug(f"Found {len(new_pci_data)} PCI records in towers.json")
        
        logger.debug("Merging PCI data (non-destructive)")
        merged_pci, new_count, updated_count = merge_pci_data(existing_pci, new_pci_data)
        
        logger.debug(f"Saving PCI database to: {output_file}")
        save_pci_database(merged_pci, output_file)
        
        total_records = len(merged_pci)
        
        logger.info(f"✓ PCI database updated successfully")
        logger.info(f"  New records added: {new_count}")
        logger.info(f"  Existing records updated: {updated_count}")
        logger.info(f"  Manually added entries: preserved")
        logger.info(f"  Total records: {total_records}")
        
        return {
            'success': True,
            'new_records': new_count,
            'updated_records': updated_count,
            'total_records': total_records,
            'output_file': str(Path(output_file).absolute())
        }
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return {
            'success': False,
            'error': str(e),
            'new_records': 0,
            'updated_records': 0,
            'total_records': 0
        }
    except ValueError as e:
        logger.error(f"Data error: {e}")
        return {
            'success': False,
            'error': str(e),
            'new_records': 0,
            'updated_records': 0,
            'total_records': 0
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            'success': False,
            'error': str(e),
            'new_records': 0,
            'updated_records': 0,
            'total_records': 0
        }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pci_generator.py <towers_file> [pci_output_file]")
        print("Example: python pci_generator.py data/tower_data/towers.json data/tower_data/pci.json")
        sys.exit(1)
    
    towers_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/tower_data/pci.json'
    
    result = populate_pci_database(towers_path, output_path, verbose=True)
    
    if result['success']:
        print(f"\n✓ Success! Output: {result['output_file']}")
        sys.exit(0)
    else:
        print(f"\n✗ Error: {result['error']}")
        sys.exit(1)