import sys
from pathlib import Path


def validate_log_file_args(log_file: str) -> None:
    if log_file is None:
        print("Usage: python main.py <log_file>")
        print("Example: python main.py lln_1.txt")
        sys.exit(1)


def validate_required_files(paths: dict) -> None:
    if not paths["input_log"].exists():
        print(f"Error: Log file not found: {paths['input_log']}")
        sys.exit(1)

    if not paths["towers_db"].exists():
        print(f"Error: Tower database not found: {paths['towers_db']}")
        sys.exit(1)

    if not paths["pci_db"].exists():
        print(f"Warning: PCI database not found: {paths['pci_db']}")


def validate_output_file(file_path: Path, description: str) -> bool:
    if not file_path.exists():
        print(f"Error: {description} failed - no output file: {file_path}")
        return False
    return True


def get_csv_row_count(file_path: Path) -> int:
    return sum(1 for _ in open(file_path)) - 1
