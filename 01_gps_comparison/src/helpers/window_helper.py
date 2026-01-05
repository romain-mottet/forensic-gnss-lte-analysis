from pathlib import Path
from typing import Dict, List
from src.find_optimal_match_window import find_optimal_window, format_results_table
from src.config.parameters import TEST_WINDOWS_PARAM
from src.helpers.print_helper import PipelineMessages


def test_windows_for_pair(
    smartphone_file: Path, 
    watch_file: Path, 
    test_windows: List[float] = None
) -> Dict:

    if test_windows is None:
        test_windows = TEST_WINDOWS_PARAM
    
    # Extract location/session from filename (for richer metadata)
    parts = smartphone_file.stem.split("_")
    location = parts[2] if len(parts) >= 3 else "unknown"
    number = parts[3] if len(parts) >= 4 else "5"
    
    try:
        # Call the original optimal window finder
        results = find_optimal_window(smartphone_file, watch_file, test_windows)
        
        # Rich metadata format
        standardized_result = {
            'file_pair': {
                'smartphone': smartphone_file,
                'watch': watch_file
            },
            'location': location,
            'session': number,
            **results  # optimal_window, best_match_rate, results, etc.
        }
        
        return standardized_result
        
    except Exception as e:
        return {
            'file_pair': {
                'smartphone': smartphone_file,
                'watch': watch_file
            },
            'location': location,
            'session': number,
            'error': str(e)
        }


def print_window_recommendations(all_results: List[Dict]) -> None:
    PipelineMessages.step2_recommendations_header()

    window_votes: Dict[float, int] = {}
    for result in all_results:
        w = result["optimal_window"]
        window_votes[w] = window_votes.get(w, 0) + 1

    most_common = max(window_votes.items(), key=lambda x: x[1])[0]
    PipelineMessages.step2_recommended_window(most_common, window_votes[most_common], len(all_results))

    PipelineMessages.step2_recommendations_by_dataset_header()
    for result in all_results:
        spfile = result["file_pair"]["smartphone"]
        w = result["optimal_window"]
        rate = result["best_match_rate"]
        PipelineMessages.step2_dataset_recommendation(spfile, w, rate)

    print()
    PipelineMessages.step2_footer()
