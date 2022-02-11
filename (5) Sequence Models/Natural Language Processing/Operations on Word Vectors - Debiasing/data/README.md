Since the file at Operations on Word Vectors - Debiasing/data/glove.6B.50d.txt is larger than 100MB, I can't really upload it to github.

I will split the file with the command: "split -b 45M glove.6B.50d.txt "parts-prefix""

To join the files, in order to run the jupyter notebook use the following command: "cat parts-prefix* > glove.6B.50d.txt"
