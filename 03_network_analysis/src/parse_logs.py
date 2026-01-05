import pandas as pd
import sys
from pathlib import Path

def parse_network_logs(input_file='data/raw_logs/lln_1.txt', output_file='data/lln_1_data/parsed_lln_1.csv'):
    
    print(f"[1/5] Reading TSV file: {input_file}")
    
    try:
        rows = []
        header = None
        max_cols = 0
        
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                # Remove line ending
                line = line.rstrip('\r\n')
                
                # Parse header (first line)
                if line_num == 1:
                    header = line.split('\t')
                    max_cols = len(header)
                    print(f" ✓ Header: {max_cols} columns")
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse data rows
                values = line.split('\t')
                
                # Pad to match header length
                if len(values) < max_cols:
                    values.extend([''] * (max_cols - len(values)))
                elif len(values) > max_cols:
                    values = values[:max_cols]
                
                rows.append(values)
        
        print(f" ✓ Read {len(rows)} data rows")
        
        print(f"\n[2/5] Creating DataFrame")
        df = pd.DataFrame(rows, columns=header)
        print(f" ✓ Shape: {df.shape}")
        
        # Validate
        print(f"\n[3/5] Validating data")
        print(f" ✓ Rows: {len(df)}")
        print(f" ✓ Columns: {len(df.columns)}")
        
        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        print(f"\n[4/5] Saving to CSV: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8')
        file_size = Path(output_file).stat().st_size
        print(f" ✓ File size: {file_size:,} bytes")
        
        # Summary
        print(f"\n[5/5] Summary")
        print(f" ✓ Input file: {input_file}")
        print(f" ✓ Output file: {output_file}")
        print(f" ✓ Rows parsed: {len(df)}")
        print(f" ✓ Columns preserved: {len(df.columns)}")
        
        # Show cell distribution if RAWCELLID exists
        if 'RAWCELLID' in df.columns:
            print(f"\n Cell distribution:")
            cell_counts = df['RAWCELLID'].value_counts()
            for cell, count in cell_counts.head(5).items():
                print(f"  - Cell {cell}: {count} records")
            if len(cell_counts) > 5:
                print(f"  - ... and {len(cell_counts) - 5} more cells")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Step 1.1 standalone"""
    print("="*80)
    print("STEP 1.1: PARSE NETWORK LOG")
    print("="*80)
    print()
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/raw_logs/lln_1.txt'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'data/lln_1_data/parsed_lln_1.csv'
    
    success = parse_network_logs(input_file, output_file)
    
    print("\n" + "="*80)
    if success:
        print("[OK] STEP 1.1 COMPLETE")
    else:
        print("[FAILED] STEP 1.1 FAILED")
    print("="*80)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())