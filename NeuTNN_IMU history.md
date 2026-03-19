# NeuTNN IMU Session Classification (4-class)

## Task
- Classes: stand / slow / fast / run
- Data source: imu_dataset_clean manifest (ses-02/03/04/05 -> class 0/1/2/3)
- Split strategy: subject-aware per-class split, then balanced window sampling

## Final Model
- File: activeDendrite_imu_eeg_v3.py
- Encoder: DeltaModulationEncoder
  - n_thresholds=3
  - threshold_base=0.0112
  - scale searched around 1.8~2.2
- Architecture:
  - L1: temporal pattern extraction (unsupervised STDP-like)
  - L2: grouped dendrite readout (supervised update)

## Key Fixes Implemented
1. Removed leakage-like behavior by splitting by subject/class before windowing.
2. Balanced window loader so all classes have equal train/test windows.
3. Added stronger L2 class-separation initialization and improved voting rule.

## Tried Methods
1. Baseline temporal NeuTNN (v2 search, 32 configs)
   - Result: collapsed to class 0, accuracy ~25%
2. Improved v3
   - Added class-biased L2 init
   - Added improved grouped voting
   - Result: **BEST ACCURACY 85.00%**

## Best Result (v3)
Confusion matrix:
[[ 83   0   0  67]
 [ 16 134   0   0]
 [  0   3 147   0]
 [  0   0   4 146]]

Per-class recall:
- Class 0: 55.3%
- Class 1: 89.3%
- Class 2: 98.0%
- Class 3: 97.3%

## Current Limitation
- Main error mode: class 0 confused with class 3.
- Next optimization should focus on stand vs run separation.