import scipy.io
import warnings

mat_file = r"C:\Users\Enzob\Downloads\somnieve_dataset_geneva20260320.mat"

print("=" * 70)
print("Inspecting MATLAB file...")
print("=" * 70)

# Suppress the duplicate variable warning for cleaner output
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    try:
        # Try to load the file
        data = scipy.io.loadmat(mat_file)
        
        print("\nAll keys in loaded data:")
        print(list(data.keys()))
        
        print("\n" + "=" * 70)
        print("Detailed information:")
        print("=" * 70)
        
        for key in sorted(data.keys()):
            if not key.startswith('__'):
                value = data[key]
                print(f"\nVariable: {repr(key)}")
                print(f"  Type: {type(value).__name__}")
                print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                
                if hasattr(value, 'dtype'):
                    print(f"  Data type: {value.dtype}")
                    
                    # If it's a structured array, show fields
                    if value.dtype.names:
                        print(f"  Fields: {value.dtype.names}")
                        for field in value.dtype.names:
                            field_data = value[field]
                            print(f"    - {field}: shape={field_data.shape}, dtype={field_data.dtype}")
                
                # Show first few elements/rows
                try:
                    if hasattr(value, 'shape'):
                        if value.size > 0:
                            if len(value.shape) == 1 and value.shape[0] > 0:
                                preview = value[0] if value.shape[0] == 1 else value[:min(3, value.shape[0])]
                                print(f"  Sample data: {preview}")
                except:
                    pass
                    
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()


