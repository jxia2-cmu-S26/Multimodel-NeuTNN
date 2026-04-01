from pathlib import Path
import mne
import json

# Check IMU sample rate from manifest
with open('imu_dataset_clean/manifest.json', 'r') as f:
    manifest = json.load(f)
    if manifest:
        imu_sfreq = manifest[0].get('sfreq')
        print(f"IMU Sample Rate: {imu_sfreq} Hz")

# Check EEG sample rate
eeg_root = Path('..') / 'osfstorage-archive'
eeg_files = list(eeg_root.glob('sub-*/ses-*/eeg/*task-ERP_eeg.vhdr'))
if eeg_files:
    eeg_file = eeg_files[0]
    print(f'Checking EEG file: {eeg_file}')
    try:
        raw = mne.io.read_raw_brainvision(str(eeg_file), preload=False, verbose=False)
        eeg_sfreq = raw.info['sfreq']
        print(f'EEG Sample Rate: {eeg_sfreq} Hz')
        
        # Calculate window lengths for 1 second duration
        window_len_imu = int(imu_sfreq * 1.0)
        window_len_eeg = int(eeg_sfreq * 1.0)
        
        print(f"\nFor 1-second windows:")
        print(f"  IMU window_len: {window_len_imu} samples")
        print(f"  EEG window_len: {window_len_eeg} samples")
        
    except Exception as e:
        print(f'Error: {e}')
else:
    print('No EEG files found')
