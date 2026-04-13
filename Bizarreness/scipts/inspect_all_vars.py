import scipy.io
import warnings
import numpy as np

mat_file = r"C:\Users\Enzob\Downloads\somnieve_dataset_geneva20260320.mat"

print("=" * 80)
print("COMPLETE FILE INSPECTION - ALL VARIABLES")
print("=" * 80)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    try:
        # Load with all variables
        data = scipy.io.loadmat(mat_file)
        
        print("\nAll keys in file:")
        for key in data.keys():
            print(f"  - {key}")
        
        print("\n" + "=" * 80)
        print("EXAMINING __function_workspace__:")
        print("=" * 80)
        
        if '__function_workspace__' in data:
            ws = data['__function_workspace__']
            print(f"\nType: {type(ws)}")
            print(f"Shape: {ws.shape if hasattr(ws, 'shape') else 'N/A'}")
            print(f"Dtype: {ws.dtype if hasattr(ws, 'dtype') else 'N/A'}")
            
            if isinstance(ws, np.ndarray):
                print(f"\nWorkspace contents:")
                if ws.dtype == object:
                    print(f"  This is an object array with {ws.size} elements")
                    for i, item in enumerate(ws.flat):
                        print(f"\n  Item {i}:")
                        print(f"    Type: {type(item).__name__}")
                        if isinstance(item, np.ndarray):
                            print(f"    Shape: {item.shape}")
                            print(f"    Dtype: {item.dtype}")
                            print(f"    Content (first 100 chars): {str(item)[:100]}")
                        elif isinstance(item, dict):
                            print(f"    Dict keys: {list(item.keys())}")
                        else:
                            print(f"    Content: {item}")
        
        print("\n" + "=" * 80)
        print("EXAMINING __globals__:")
        print("=" * 80)
        
        if '__globals__' in data:
            globals_var = data['__globals__']
            print(f"\nType: {type(globals_var)}")
            print(f"Shape: {globals_var.shape if hasattr(globals_var, 'shape') else 'N/A'}")
            
            if isinstance(globals_var, np.ndarray) and globals_var.size > 0:
                print(f"Globals: {globals_var}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
