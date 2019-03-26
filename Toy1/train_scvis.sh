# TODO:  use 9d data

cd ../scvis-dev
scvis train --data_matrix_file ./data/synthetic_2d_2200.tsv --out_dir ./output/ --data_label_file ./data/synthetic_2d_2200_label.tsv --verbose --verbose_interval 50
mv output ../Toy1/output
