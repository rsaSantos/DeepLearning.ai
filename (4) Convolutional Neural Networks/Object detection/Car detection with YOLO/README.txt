Since the file at Car detection with YOLO/model_data/variables/variables.data-00000-of-00001 is larger than 100MB, I can't really upload it to github.

I will split the file into two with the command: "split -b 45M variables.data-00000-of-00001 "parts-prefix""
To join the files, in order to run the jupyter notebook use the following command: "cat parts-prefix* > variables.data-00000-of-00001"
