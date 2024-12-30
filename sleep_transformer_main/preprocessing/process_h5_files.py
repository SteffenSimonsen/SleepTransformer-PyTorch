import h5py
import numpy as np
import os
import argparse
import json
from create_spectrograms import create_spectrogram_images



def get_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        unique_values = None
        if name.endswith('/hypnogram'):
            unique, counts = np.unique(obj[()], return_counts=True)
            unique_values = {int(k): int(v) for k, v in zip(unique, counts)}
        return {
            "type": "Dataset",
            "name": name,
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "unique_values": unique_values
        }
    elif isinstance(obj, h5py.Group):
        return {
            "type": "Group",
            "name": name,
            "children": []
        }
    else:
        return None

def convert_and_inspect_hdf5(input_file):
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_spectro{ext}"
    json_file = f"{base}_spectro_structure.json"
    
    structure = {"name": os.path.basename(output_file), "children": []}
    
    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        data = f_in['data']
        data_out = f_out.create_group('data')
        
        #subjects = [key for key in data.keys() if key.startswith('sub-')]
        subjects = [key for key in data.keys() if key != 'meta']
        total_sessions = 0
        
        for subject_key in subjects:
            subject = data[subject_key]
            subject_out = data_out.create_group(subject_key)
            
            for session_key in subject.keys():
                session = subject[session_key]
                session_out = subject_out.create_group(session_key)
                total_sessions += 1
                
                
                labels = session['hypnogram'][:]
                session_out.create_dataset('hypnogram', data=labels)
                
                psg_out = session_out.create_group('psg')
                for channel_key in session['psg'].keys():
                    channel = session['psg'][channel_key][:]
                    _, _, spectrograms = create_spectrogram_images(channel, 128)
                    
                    
                    psg_out.create_dataset(channel_key, data=spectrograms)
                
                print(f"Processed: {subject_key}/{session_key}")
        
        
        def visit_func(name, obj):
            item = get_hdf5_structure(name, obj)
            if item:
                current = structure
                for part in name.split('/'):
                    if part:
                        for child in current["children"]:
                            if child["name"] == part:
                                current = child
                                break
                        else:
                            new_item = {"name": part, "children": []}
                            current["children"].append(new_item)
                            current = new_item
                current.update(item)
        
        f_out.visititems(visit_func)
        
        
        sample_subject = subjects[0]
        sample_session = list(f_out['data'][sample_subject].keys())[0]
        psg_channels = list(f_out['data'][sample_subject][sample_session]['psg'].keys())
        sample_channel = psg_channels[0]
        sample_shape = f_out['data'][sample_subject][sample_session]['psg'][sample_channel].shape
        
        
        hypnogram = f_out['data'][sample_subject][sample_session]['hypnogram'][()]
        unique_labels = sorted([int(label) for label in np.unique(hypnogram)])
        
        structure["summary"] = {
            "total_subjects": len(subjects),
            "total_sessions": total_sessions,
            "psg_channels": psg_channels,
            "sample_channel_shape": list(sample_shape),
            "unique_labels": unique_labels
        }
    
    
    with open(json_file, 'w') as out_file:
        json.dump(structure, out_file, indent=2)
    
    print(f"Conversion complete. Output saved to: {output_file}")
    print(f"Structure information saved to: {json_file}")

def process_directory(directory_path):
    h5_files = [f for f in os.listdir(directory_path) 
            if (f.endswith('.h5') or f.endswith('.hdf5')) 
            and 'spectro' not in f]
    
    if not h5_files:
        print(f"No .h5 or .hdf5 files found in {directory_path}")
        return

    print(f"Found {len(h5_files)} file(s) to process.")

    for filename in h5_files:
        full_path = os.path.join(directory_path, filename)
        print(f"Processing {filename}...")

        try:
            convert_and_inspect_hdf5(full_path)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    print("All files processed.")

def main():
    parser = argparse.ArgumentParser(description='Convert H5 files to spectrograms and save structure information.')
    parser.add_argument('directory', type=str, help='Path to the directory containing H5 files')
    args = parser.parse_args()

    process_directory(args.directory)

if __name__ == "__main__":
    main()
