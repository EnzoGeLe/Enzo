import scipy.io
import h5py

mat_file = r"C:\Users\Enzob\Downloads\somnieve_dataset_geneva20260320.mat"

print("=" * 70)
print("Attempting to inspect .mat file...")
print("=" * 70)

# First, try to open as HDF5 (newer MATLAB format)
try:
    print("\n--- Using HDF5 format ---")
    with h5py.File(mat_file, 'r') as f:
        print("Top-level keys in HDF5 file:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: Dataset, shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  {name}: Group")
        
        for key in f.keys():
            print(f"\n{key}:")
            f[key].visititems(print_structure)
            
except Exception as e:
    print(f"HDF5 approach failed: {e}\n")
    
    # Try the regular scipy.io.loadmat approach
    print("--- Using scipy.io.loadmat ---")
    try:
        data = scipy.io.loadmat(mat_file)
        
        print("\nVariables in the .mat file:")
        for key in sorted(data.keys()):
            if not key.startswith('__'):
                value = data[key]
                print(f"\n{key}:")
                print(f"  Type: {type(value).__name__}")
                
                if hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                    print(f"  Dtype: {value.dtype}")
                
                if hasattr(value, 'dtype') and value.dtype.names:
                    print(f"  Field names: {value.dtype.names}")
                
    except Exception as e2:
        print(f"scipy.io.loadmat failed: {e2}")

