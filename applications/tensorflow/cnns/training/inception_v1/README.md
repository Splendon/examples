Run inception_v1 model on Graphcore IPU.

1.make tf-record
python3 create_tf_record.py

2.train inception_v1 model on cpu
python3 inception_v1_cpu_train_val.py
