from scipy.io.matlab import varmats_from_mat
import scipy.io
import os
import warnings

mat_file = r"C:\Users\Enzob\Downloads\somnieve_dataset_geneva20260320.mat"

print("=" * 80)
print("SPLITTING .MAT FILE INTO SEPARATE VARIABLE FILES")
print("=" * 80)

try:
    print(f"\nSplitting {mat_file}...")
    
    # Split the file (varmats_from_mat expects a file object)
    with open(mat_file, 'rb') as mat_stream:
        variables = {}
        for var_name, var_file in varmats_from_mat(mat_stream):
            # Skip empty variable names
            if not var_name:
                print(f"\nSkipping empty variable name")
                continue
                
            print(f"\nVariable: {repr(var_name)}")
            
            # Read the variable from the temp file
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = scipy.io.loadmat(var_file)
            
            var_data = data[var_name]
            
            print(f"  Type: {type(var_data).__name__}")
            if hasattr(var_data, 'shape'):
                print(f"  Shape: {var_data.shape}")
                print(f"  Dtype: {var_data.dtype if hasattr(var_data, 'dtype') else 'N/A'}")
            
            if hasattr(var_data, 'dtype') and var_data.dtype.names:
                print(f"  Structured dtype fields: {var_data.dtype.names}")
                # For MatlabOpaque, show the table name
                if 's0' in var_data.dtype.names:
                    table_name = var_data[0]['s0']
                    print(f"  Table name: {table_name}")
            
            # Show preview
            if hasattr(var_data, 'shape'):
                if var_data.size <= 20:
                    print(f"  Content: {var_data}")
                else:
                    print(f"  Size: {var_data.size} elements")
                    # For opaque MATLAB tables, show the data array
                    if hasattr(var_data, 'dtype') and 'arr' in var_data.dtype.names:
                        arr = var_data[0]['arr']
                        print(f"  Table data shape: {arr.shape}")
                        print(f"  Table data:\n{arr}")
            
            variables[var_name] = var_data
        
        print("\n" + "=" * 80)
        print(f"TOTAL VARIABLES FOUND: {len(variables)}")
        print("=" * 80)
                
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
