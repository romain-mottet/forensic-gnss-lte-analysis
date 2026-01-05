import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

def extract_neighbor_cells(df):
    neighbor_cells = {}
    
    # Neighbor cell columns: NCell1 through NCell18
    for i in range(1, 19):
        col_name = f'NCell{i}'
        if col_name in df.columns:
            neighbors = df[col_name].dropna()
            for pci_val in neighbors:
                if pd.notna(pci_val) and str(pci_val).strip():
                    try:
                        pci = str(int(float(pci_val)))
                        neighbor_cells[pci] = neighbor_cells.get(pci, 0) + 1
                    except (ValueError, TypeError):
                        pass
    return neighbor_cells

def verify_database_coverage(log_file='data/parsed_data/parsed_logs.csv', 
                           towers_file='data/tower_data/towers.json',
                           pci_file='data/tower_data/pci.json',
                           report_file='reports/database_coverage_report.txt',
                           status_file='reports/towers_status.json'):
    
    print(f"[1/7] Loading towers database: {towers_file}")
    
    try:
        # Load towers database
        with open(towers_file, 'r', encoding='utf-8') as f:
            towers_db = json.load(f)
        print(f"      [OK] Towers in database: {len(towers_db)}")
        
        # Load PCI database
        print(f"\n[2/7] Loading PCI lookup database: {pci_file}")
        pci_db = {}
        if Path(pci_file).exists():
            with open(pci_file, 'r', encoding='utf-8') as f:
                pci_db = json.load(f)
            print(f"      [OK] PCIs in database: {len(pci_db)}")
        else:
            print(f"      [!!] PCI database not found. Generating from towers.json...")
            # Auto-generate from towers
            for rawcellid, tower_info in towers_db.items():
                pci = str(tower_info.get('pci', 'N/A'))
                if pci != 'N/A' and pci not in pci_db:
                    pci_db[pci] = {
                        'pci': pci,
                        'latitude': tower_info.get('latitude', 'N/A'),
                        'longitude': tower_info.get('longitude', 'N/A')
                    }
            print(f"      [OK] Generated {len(pci_db)} PCI entries")

        # Load logs
        print(f"\n[3/7] Loading parsed logs: {log_file}")
        df = pd.read_csv(log_file, encoding='utf-8')
        print(f"      [OK] Measurements: {len(df)}")
        
        # Extract serving cells
        print(f"\n[4/7] Extracting SERVING cell IDs")
        raw_cells = df['RAWCELLID'].unique()
        serving_cells = []
        cell_details = {}
        
        for cell in raw_cells:
            if pd.notna(cell):
                cell_str = str(int(cell)) if isinstance(cell, (int, float)) else str(cell)
                serving_cells.append(cell_str)
                
                # Get one row for details
                cell_mask = df['RAWCELLID'].astype(str) == cell_str
                cell_rows = df[cell_mask]
                count = int(cell_mask.sum())
                
                first_row = cell_rows.iloc[0]
                pci = first_row.get('PSC', 'N/A')
                pci = str(int(pci)) if pd.notna(pci) else 'N/A'
                band = first_row.get('Band', 'N/A')
                band = str(band) if pd.notna(band) else 'N/A'
                operator_code = first_row.get('Operator', 'N/A')
                operator_code = str(int(operator_code)) if pd.notna(operator_code) else 'N/A'
                network_tech = first_row.get('NetworkTech', 'N/A')
                network_tech = str(network_tech) if pd.notna(network_tech) else 'N/A'
                node = first_row.get('Node', 'N/A')
                node = str(int(node)) if pd.notna(node) else 'N/A'
                
                cell_details[cell_str] = {
                    'measurements': count,
                    'pci': pci,
                    'band': band,
                    'operator_code': operator_code,
                    'network_tech': network_tech,
                    'node': node
                }
        
        serving_cells = sorted(set(serving_cells))
        print(f"      [OK] Unique SERVING cells: {len(serving_cells)}")
        
        # Extract neighbor cells
        print(f"\n[5/7] Extracting NEIGHBOR cell PCIs")
        neighbor_pci_counts = extract_neighbor_cells(df)
        print(f"      [OK] Unique NEIGHBOR PCIs: {len(neighbor_pci_counts)}")
        
        # Cross-reference SERVING cells
        print(f"\n[6/7] Cross-referencing cells with databases")
        serving_found = []
        serving_missing = []
        
        for cell_id in serving_cells:
            if str(cell_id) in towers_db:
                serving_found.append(cell_id)
                print(f"      [OK] SERVING {cell_id}: Found in towers.json")
            else:
                serving_missing.append(cell_id)
                print(f"      [!!] SERVING {cell_id}: MISSING")
                
        # Cross-reference NEIGHBOR cells (check both databases)
        neighbor_found_towers = []
        neighbor_found_pci = []
        neighbor_missing = []
        
        # Build PCI->RAWCELLID map from towers
        pci_to_rawcellid = {}
        for rawcellid, tower_info in towers_db.items():
            pci = str(tower_info.get('pci', 'N/A'))
            if pci != 'N/A':
                if pci not in pci_to_rawcellid:
                    pci_to_rawcellid[pci] = []
                pci_to_rawcellid[pci].append(rawcellid)
        
        for pci in sorted(neighbor_pci_counts.keys()):
            if pci in pci_to_rawcellid:
                neighbor_found_towers.append(pci)
                print(f"      [OK] NEIGHBOR PCI {pci}: Found in towers.json")
            elif pci in pci_db:
                neighbor_found_pci.append(pci)
                print(f"      [OK] NEIGHBOR PCI {pci}: Found in pci.json")
            else:
                neighbor_missing.append(pci)
                print(f"      [!!] NEIGHBOR PCI {pci}: MISSING from both databases")

        # Calculate statistics
        print(f"\n[7/7] Generating reports")
        
        serving_coverage = 100 * len(serving_found) / len(serving_cells) if serving_cells else 0
        
        neighbor_found_total = len(neighbor_found_towers) + len(neighbor_found_pci)
        neighbor_total = len(neighbor_pci_counts)
        neighbor_coverage_towers = 100 * len(neighbor_found_towers) / neighbor_total if neighbor_total else 0
        neighbor_coverage_pci = 100 * len(neighbor_found_pci) / neighbor_total if neighbor_total else 0
        neighbor_coverage = 100 * neighbor_found_total / neighbor_total if neighbor_total else 100
        
        overall_total = len(serving_cells) + neighbor_total
        overall_found = len(serving_found) + neighbor_found_total
        overall_missing = len(serving_missing) + len(neighbor_missing)
        overall_coverage = 100 * overall_found / overall_total if overall_total > 0 else 100

        # --- CSV REPORT GENERATION ---
        csv_file = str(Path(report_file).with_suffix('.csv'))
        
        metrics_data = {
            'timestamp': [datetime.now().isoformat()],
            'log_file': [Path(log_file).name],
            'total_cells': [overall_total],
            'found_cells': [overall_found],
            'missing_cells': [overall_missing],
            'overall_coverage_pct': [round(overall_coverage, 2)],
            
            'serving_total': [len(serving_cells)],
            'serving_found': [len(serving_found)],
            'serving_missing': [len(serving_missing)],
            'serving_coverage_pct': [round(serving_coverage, 2)],
            
            'neighbor_total': [neighbor_total],
            'neighbor_found_total': [neighbor_found_total],
            'neighbor_found_towers': [len(neighbor_found_towers)],
            'neighbor_found_pci': [len(neighbor_found_pci)],
            'neighbor_missing': [len(neighbor_missing)],
            'neighbor_coverage_pct': [round(neighbor_coverage, 2)]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(csv_file, index=False)
        print(f"      [OK] Saved metrics CSV: {csv_file}")
        
        # --- TXT REPORT GENERATION ---
        report_lines = []
        report_lines.append("="*90)
        report_lines.append("STEP 1.2: DATABASE COVERAGE VERIFICATION (towers.json + pci.json)")
        report_lines.append("="*90)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\n{'-'*90}")
        report_lines.append("SUMMARY")
        report_lines.append(f"{'-'*90}")
        
        report_lines.append(f"\n[SERVING CELLS]")
        report_lines.append(f"  Total in log: {len(serving_cells)}")
        report_lines.append(f"  Found:        {len(serving_found)}")
        report_lines.append(f"  Missing:      {len(serving_missing)}")
        report_lines.append(f"  Coverage:     {serving_coverage:.1f}%")
        
        report_lines.append(f"\n[NEIGHBOR CELLS]")
        report_lines.append(f"  Total unique PCIs:    {neighbor_total}")
        report_lines.append(f"  Found in towers.json: {len(neighbor_found_towers)}")
        report_lines.append(f"  Found in pci.json:    {len(neighbor_found_pci)}")
        report_lines.append(f"  Missing from both:    {len(neighbor_missing)}")
        report_lines.append(f"  Coverage:             {neighbor_coverage:.1f}%")
        
        report_lines.append(f"\n[OVERALL]")
        report_lines.append(f"  Total cells:      {overall_total}")
        report_lines.append(f"  Found:            {overall_found}")
        report_lines.append(f"  Missing:          {overall_missing}")
        report_lines.append(f"  Database coverage: {overall_coverage:.1f}%")
        
        # Detailed findings
        report_lines.append(f"\n{'-'*90}")
        report_lines.append("DETAILED FINDINGS")
        report_lines.append(f"{'-'*90}")
        
        report_lines.append(f"\n[1] SERVING CELLS FOUND ({len(serving_found)}):")
        if serving_found:
            report_lines.append(f"  {'RAWCELLID':<15} {'PCI':<8} {'Band':<10} {'Op':<8} {'Tech':<8} {'Node':<12}")
            report_lines.append(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*12}")
            for cell_id in serving_found:
                details = cell_details[cell_id]
                report_lines.append(
                    f"  {cell_id:<15} {details['pci']:<8} {details['band']:<10} "
                    f"{details['operator_code']:<8} {details['network_tech']:<8} {details['node']:<12}"
                )
        else:
            report_lines.append("  (None)")
            
        report_lines.append(f"\n{'-'*90}")
        report_lines.append(f"[2] SERVING CELLS MISSING ({len(serving_missing)}):")
        if serving_missing:
            for cell_id in serving_missing:
                details = cell_details[cell_id]
                report_lines.append(f"\n  RAWCELLID {cell_id}:")
                report_lines.append(f"    PCI: {details['pci']}")
                report_lines.append(f"    Band: {details['band']}")
                report_lines.append(f"    Status: [!!] MISSING from towers.json")
        else:
            report_lines.append("  None - all serving cells found! [OK]")
            
        report_lines.append(f"\n{'-'*90}")
        report_lines.append(f"[3] NEIGHBOR CELLS FOUND IN towers.json ({len(neighbor_found_towers)}):")
        if neighbor_found_towers:
            report_lines.append(f"  {'PCI':<8} {'Measurements':<15} {'Coverage'}")
            report_lines.append(f"  {'-'*8} {'-'*15} {'-'*30}")
            for pci in neighbor_found_towers:
                count = neighbor_pci_counts[pci]
                report_lines.append(f"  {pci:<8} {count:<15} ✓ Complete tower data available")
        else:
            report_lines.append("  (None)")
            
        report_lines.append(f"\n{'-'*90}")
        report_lines.append(f"[4] NEIGHBOR CELLS FOUND IN pci.json ({len(neighbor_found_pci)}):")
        if neighbor_found_pci:
            report_lines.append(f"  {'PCI':<8} {'Measurements':<15} {'Location'}")
            report_lines.append(f"  {'-'*8} {'-'*15} {'-'*30}")
            for pci in neighbor_found_pci:
                count = neighbor_pci_counts[pci]
                pci_info = pci_db.get(pci, {})
                lat = pci_info.get('latitude', 'N/A')
                lon = pci_info.get('longitude', 'N/A')
                report_lines.append(f"  {pci:<8} {count:<15} ({lat}, {lon})")
        else:
            report_lines.append("  (None)")
            
        report_lines.append(f"\n{'-'*90}")
        report_lines.append(f"[5] NEIGHBOR CELLS MISSING ({len(neighbor_missing)}):")
        if neighbor_missing:
            report_lines.append(f"  PCIs without location data: {', '.join(neighbor_missing)}")
            report_lines.append(f"\n  Action: Research and add to towers.json, then regenerate pci.json")
        else:
            report_lines.append("  None - all neighbors have location data! [OK]")
            
        # Recommendations
        report_lines.append(f"\n{'-'*90}")
        report_lines.append("RECOMMENDATIONS")
        report_lines.append(f"{'-'*90}")
        
        if overall_coverage == 100:
            report_lines.append(f"\n[OK] 100% DATABASE COVERAGE")
            report_lines.append(f"\nAll cells (serving + neighbors) have location data.")
            report_lines.append(f"Proceed to next steps.")
        else:
            report_lines.append(f"\n[!!] {overall_coverage:.1f}% DATABASE COVERAGE")
            
            if serving_missing:
                report_lines.append(f"\nSERVING CELLS: {len(serving_missing)} missing")
                report_lines.append(f"  Action: Add to towers.json, then regenerate pci.json")
            
            if neighbor_missing:
                report_lines.append(f"\nNEIGHBOR CELLS: {len(neighbor_missing)} missing")
                report_lines.append(f"  PCIs without location: {', '.join(neighbor_missing)}")
                report_lines.append(f"  Action: Research cells, add to towers.json")
                report_lines.append(f"          Then run: python pci_generator.py")
        
        report_lines.append(f"\n{'-'*90}")
        report_lines.append("END OF REPORT")
        report_lines.append("="*90)
        
        report_text = '\n'.join(report_lines)
        print(f"\n{report_text}")
        
        # Save report
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n      [OK] Saved: {report_file}")
        
        # Save JSON status
        status = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'serving_cells': {
                    'total': len(serving_cells),
                    'found': len(serving_found),
                    'missing': len(serving_missing),
                    'coverage_percentage': round(serving_coverage, 1)
                },
                'neighbor_cells': {
                    'total': len(neighbor_pci_counts),
                    'found_in_towers': len(neighbor_found_towers),
                    'found_in_pci': len(neighbor_found_pci),
                    'missing': len(neighbor_missing),
                    'coverage_percentage': round(neighbor_coverage, 1)
                },
                'overall': {
                    'total': overall_total,
                    'found': overall_found,
                    'missing': overall_missing,
                    'coverage_percentage': round(overall_coverage, 1)
                }
            },
            'databases_used': {
                'towers': towers_file,
                'pci': pci_file
            },
            'neighbor_cells_found_in_towers': neighbor_found_towers,
            'neighbor_cells_found_in_pci': neighbor_found_pci,
            'neighbor_cells_missing': neighbor_missing
        }
        
        Path(status_file).parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        print(f"      [OK] Saved: {status_file}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*90)
    print("STEP 1.2: VERIFY TOWER DATABASE COVERAGE (towers.json + pci.json)")
    print("="*90)
    print()
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'data/parsed_data/parsed_logs.csv'
    towers_file = sys.argv[2] if len(sys.argv) > 2 else 'data/tower_data/towers.json'
    pci_file = sys.argv[3] if len(sys.argv) > 3 else 'data/tower_data/pci.json'
    report_file = sys.argv[4] if len(sys.argv) > 4 else 'reports/database_coverage_report.txt'
    status_file = sys.argv[5] if len(sys.argv) > 5 else 'reports/towers_status.json'
    
    success = verify_database_coverage(log_file, towers_file, pci_file, report_file, status_file)
    
    print("\n" + "="*90)
    if success:
        print("[OK] STEP 1.2 COMPLETE")
    else:
        print("[FAILED] STEP 1.2 FAILED")
    print("="*90)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
