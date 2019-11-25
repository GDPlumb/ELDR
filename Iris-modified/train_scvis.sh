rm -rf Model

python ../scvis/scvis train --data_matrix_file ../Iris/Data/X_corrupted.tsv --out_dir ./Model/ --verbose --verbose_interval 50
