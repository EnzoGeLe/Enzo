import h5py
import numpy as np
import pandas as pd
import warnings

mat_file = r"C:\Users\Enzob\Downloads\somnieve_dataset_geneva20260320.mat"

print("=" * 80)
print("ATTEMPTING HDF5 LOAD (MATLAB 7.3+ format)")
print("=" * 80)

try:
    with h5py.File(mat_file, 'r') as f:
        print("\n1. ROOT GROUPS/DATASETS:")
        
        def print_tree(name, obj, depth=0):
            indent = "  " * depth
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: Dataset shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}: Group")
        
        for key in f.keys():
            print_tree(key, f[key])
            f[key].visititems(lambda n, o: print_tree(n, o, 1))
        
        print("\n" + "=" * 80)
        print("2. EXTRACTING WAKE_TABLE DATA:")
        print("=" * 80)
        
        # Try common MATLAB table structure paths
        possible_paths = [
            '#refs#',
            'wake_table',
            '#MATLAB_class#',
            '#MATLAB_struct#'
        ]
        
        def explore_group(g, path=""):
            print(f"\nExploring: {path if path else 'root'}")
            for key in g.keys():
                item = g[key]
                if isinstance(item, h5py.Group):
                    explore_group(item, f"{path}/{key}" if path else key)
                elif isinstance(item, h5py.Dataset):
                    print(f"  Dataset: {key} -> shape={item.shape}, dtype={item.dtype}")
                    if item.size < 100:
                        print(f"    Content: {item[()]}")
        
        explore_group(f)
        
except FileNotFoundError:
    print(f"File not found: {mat_file}")
except Exception as e:
    print(f"Error with HDF5: {e}")
    print("\nTrying alternative approach with scipy...")
    
    import scipy.io
    
    # Try extracting with different parameters
    data = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    
    print("\nVariables (with simplify_cells=True):")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(data[key]).__name__}")
