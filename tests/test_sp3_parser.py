import sys
sys.path.append('src')
from parsers.sp3_parser import parse_sp3
import pandas as pd

def test_parse_sp3_epochs():
    """
    Test the parse_sp3 function by checking coordinates for selected satellites
    across different epochs.
    """
    # Path to the SP3 file
    file_path = 'data/COD0MGXFIN_20220010000_01D_05M_ORB.SP3'
    
    # Parse the SP3 file
    df = parse_sp3(file_path)
    
    # Ensure the DataFrame is not empty
    assert not df.empty, "DataFrame should not be empty"
    
    # Get unique epochs
    unique_epochs = df['epoch'].unique()
    
    # Select first 3-4 epochs
    selected_epochs = unique_epochs[-3:-1]  # Last 3 epochs, excluding the very last one for testing
    
    # Selected satellite, e.g., GPS satellite G04
    selected_sat = 'C45'
    
    # Dictionary to store results
    results = {}
    
    for epoch in selected_epochs:
        # Filter for the specific epoch and satellite
        coords = df[(df['epoch'] == epoch) & (df['satID'] == selected_sat)]
        
        if not coords.empty:
            results[str(epoch)] = {
                'X': coords['X'].values[0],
                'Y': coords['Y'].values[0],
                'Z': coords['Z'].values[0]
            }
        else:
            results[str(epoch)] = None  # If satellite not found in that epoch
    
    # Print the results
    print("Coordinates for selected satellite across epochs:")
    for epoch, coords in results.items():
        print(f"Epoch: {epoch}, Coords: {coords}")
    return results

if __name__ == '__main__':
    test_parse_sp3_epochs()