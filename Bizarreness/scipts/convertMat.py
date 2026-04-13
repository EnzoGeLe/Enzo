import numpy as np
import pandas as pd
import scipy.io
from scipy.io.matlab import varmats_from_mat
import warnings

mat_file = r"C:\Users\Enzob\Downloads\somnieve_dataset_geneva20260320.mat"

print("=" * 80)
print("EXTRACTING ALL TABLES FROM .MAT FILE")
print("=" * 80)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    with open(mat_file, 'rb') as mat_stream:
        table_count = 0
        for var_name, var_file in varmats_from_mat(mat_stream):
            if not var_name:
                continue
            
            data = scipy.io.loadmat(var_file)
            var_data = data[var_name]
            
            # Extract table metadata and data
            if hasattr(var_data, 'dtype') and 'arr' in var_data.dtype.names:
                matlab_table = var_data[0]
                table_name_bytes = matlab_table['s0']
                # Handle both bytes and numpy array cases
                if isinstance(table_name_bytes, bytes):
                    table_name = table_name_bytes.decode('utf-8', errors='ignore')
                else:
                    table_name = str(table_name_bytes)
                arr = matlab_table['arr']
                
                print(f"\n{'='*80}")
                print(f"Table: {table_name}")
                print(f"Shape: {arr.shape}")
                print(f"Data type: {arr.dtype}")
                print(f"{'='*80}")
                
                # Create DataFrame and save
                df = pd.DataFrame(arr)
                output_file = f"{table_name}.csv"
                df.to_csv(output_file, index=False, header=False)
                
                print(f"Saved to: {output_file}")
                print(f"\nData preview:")
                print(df)
                
                table_count += 1

print(f"\n{'='*80}")
print(f"Total tables extracted: {table_count}")
print("="*80)
