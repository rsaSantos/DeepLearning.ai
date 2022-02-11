Since the files at Trigger Word Detection/XY_train/X.npy and Trigger Word Detection/XY_dev/X_dev.npy are larger than 100MB, I can't really upload it to github.

I will split the files with the command: "split -b 45M X.npy "parts-prefix"" and "split -b 45M X_dev.npy "parts-prefix""
To join the files, in order to run the jupyter notebook use the following command: "cat parts-prefix* > X.npy" and "cat parts-prefix* > X_dev.npy"
