Run inception_v1 model on Graphcore IPU.

1.make tf-record

python3 create_tf_record.py

2.train inception_v1 model on cpu

1)test inception_v1 model on cpu with synthetic tensor input

python3 inception_v1_input.py

2)train model on cpu

python3 inception_v1_cpu_train_val.py

3.train inception_v1 model on ipu

1)test inception_v1 model on ipu with synthetic tensor input

python3 inception_v1_ipu_input.py

2)train model on ipu

python3 ipu_train_simple.py
