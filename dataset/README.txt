Training dataset (SYNTHETIC demo data, not real patients)

40 subjects, 20 labelled MDD (1) and 20 healthy (0). One file per subject:
  subject_000.npy ... subject_039.npy   NumPy arrays, shape (19 channels, 11250 samples)
                                        250 Hz, 45 seconds, microvolts, 10-20 montage, stored as float16
  labels.csv                            columns: file, subject, label

Channel order: Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2

The file named SYNTHETIC marks this as demo data. While it exists, the dashboard shows the
Demo badge and banner. Delete it when you replace these files with real recordings.

Regenerate this set with:               python gen_data.py
Longer recordings (90 s, larger files): DUR=90 python gen_data.py     (Windows PowerShell: $env:DUR=90; python gen_data.py)
After changing the dataset, retrain with: python train.py
