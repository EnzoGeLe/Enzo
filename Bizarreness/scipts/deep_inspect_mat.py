import scipy.io
import warnings
import numpy as np

mat_file = r"C:\Users\Enzob\Downloads\somnieve_dataset_geneva20260320.mat"

print("=" * 80)
print("DEEP INSPECTION OF MATLAB TABLE STRUCTURE")
print("=" * 80)

# Suppress the duplicate variable warning for cleaner output
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    try:
        # Load the file
        data = scipy.io.loadmat(mat_file)
        
        print("\n1. TOP-LEVEL KEYS:")
        print(list(data.keys()))
        
        print("\n" + "=" * 80)
        print("2. DETAILED STRUCTURE OF 'None' VARIABLE:")
        print("=" * 80)
        
        matlab_table = data['None']
        print(f"\nType: {type(matlab_table)}")
        print(f"Shape: {matlab_table.shape}")
        print(f"Dtype: {matlab_table.dtype}")
        
        # Get the first (and only) element
        table_element = matlab_table[0]
        print(f"\nElement 0 type: {type(table_element)}")
        print(f"Element 0 is numpy.void: {isinstance(table_element, np.void)}")
        
        # Access all fields
        print(f"\nFields in table_element: {matlab_table.dtype.names}")
        
        for field_name in matlab_table.dtype.names:
            field_data = table_element[field_name]
            print(f"\n--- Field: '{field_name}' ---")
            print(f"  Type: {type(field_data)}")
            print(f"  Content: {field_data}")
            
            if isinstance(field_data, np.ndarray):
                print(f"  Shape: {field_data.shape}")
                print(f"  Dtype: {field_data.dtype}")
                
                # If it's an object array, inspect its contents
                if field_data.dtype == object:
                    print(f"  Object array contents:")
                    for i, obj in enumerate(field_data.flat[:5]):  # Show first 5
                        print(f"    [{i}] {type(obj).__name__}: {obj if not isinstance(obj, np.ndarray) else f'array shape {obj.shape}'}")
                        if isinstance(obj, np.ndarray) and obj.size <= 20:
                            print(f"        -> {obj}")
        
        # Specifically examine the 'arr' field more deeply
        print("\n" + "=" * 80)
        print("3. DETAILED ANALYSIS OF 'arr' FIELD:")
        print("=" * 80)
        
        arr_field = table_element['arr']
        print(f"\narr_field type: {type(arr_field)}")
        print(f"arr_field shape: {arr_field.shape}")
        print(f"arr_field dtype: {arr_field.dtype}")
        
        if arr_field.dtype == object and arr_field.size > 0:
            print(f"\nContents of arr_field (object array):")
            for i, item in enumerate(arr_field.flat):
                print(f"\n  Item {i}:")
                print(f"    Type: {type(item)}")
                if isinstance(item, np.ndarray):
                    print(f"    Shape: {item.shape}")
                    print(f"    Dtype: {item.dtype}")
                    print(f"    Content preview:\n{item}")
                else:
                    print(f"    Content: {item}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
