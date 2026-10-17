Sample EEG files to upload in the dashboard (SYNTHETIC demo data, not real patients)

Each file: 19 channels (10-20 montage), 250 Hz, 45 seconds. The model never saw these recordings.

healthy_sample_1..3.npy   recordings simulated without a depression-like pattern
mdd_sample_1..3.npy       recordings simulated with a depression-like pattern
healthy_sample_1.csv, mdd_sample_1.csv   the same kind of data as CSV (one row per sample, one column per channel, header row)

The pattern is deliberately weak and varies between subjects, so a few recordings sit close to the
decision threshold and the model gets some wrong. That mirrors the model's overall accuracy.

Channel order: Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2
