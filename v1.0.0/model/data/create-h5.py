import numpy as np
import h5py
import os

data_dir = "/home/steff/SleepTransformer/model/data/DataSpectrogramsNew"
output_file = '/home/steff/SleepTransformer/model/data/all_subjects.h5'
chunk_size = 1000  # Adjust based on your memory constraints

subject_count = 0

with h5py.File(output_file, 'w') as hf:
    for filename in os.listdir(data_dir):
        if filename.endswith('_sleep_stages.npy'):
            subject_id = filename.split('_')[0]
            subject_count += 1
            
            one_hot_stages = np.load(os.path.join(data_dir, filename))
            spectrograms = np.load(os.path.join(data_dir, f'{subject_id}_spectrograms.npy'))[0]

            combined_stages = np.column_stack((
                one_hot_stages[:, :4],
                np.sum(one_hot_stages[:, 4:], axis=1, keepdims=True)
            ))

            subject_group = hf.create_group(subject_id)
            stages_dataset = subject_group.create_dataset('stages', 
                                                          shape=(0, combined_stages.shape[1]), 
                                                          maxshape=(None, combined_stages.shape[1]), 
                                                          chunks=True)
            spectrograms_dataset = subject_group.create_dataset('spectrograms', 
                                                                shape=(0, *spectrograms.shape[1:]), 
                                                                maxshape=(None, *spectrograms.shape[1:]), 
                                                                chunks=True)

            for i in range(0, len(combined_stages), chunk_size):
                stages_chunk = combined_stages[i:i+chunk_size]
                spectrograms_chunk = spectrograms[i:i+chunk_size]

                stages_dataset.resize(stages_dataset.shape[0] + len(stages_chunk), axis=0)
                stages_dataset[-len(stages_chunk):] = stages_chunk

                spectrograms_dataset.resize(spectrograms_dataset.shape[0] + len(spectrograms_chunk), axis=0)
                spectrograms_dataset[-len(spectrograms_chunk):] = spectrograms_chunk

    hf.create_dataset('subject_ids', data=np.array(list(hf.keys())[:-1], dtype='S'))

print(f"Data saved to {output_file}")
print(f"Number of subjects: {subject_count}")
