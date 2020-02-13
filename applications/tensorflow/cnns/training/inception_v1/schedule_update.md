First week：2/10 - 2/14
A.项目开展整体思路
1.从模型参数上理论计算能不能把整个inception_v1 model放在ipu上；  --> 可以
2.synthetic tensor输入inception_v1(cpu --> ipu)
3.结合train.py把inception_v1编译进单卡ipu；
4.验证train.py在单卡ipu训练上loss是否收敛；
5.在loss可以收敛的前提下，单卡调优SOTA；
6.开展多卡工作+调优SOTA；

B.关于模型的选型
训练时显存的占用，主要是Model本身和每一层的output的总和。
inception_v1用的是momentum optimizer
在momentum训练下：ram = model * 3 + batch_size * lapyers_output * 2 # fowward and backward
根据理论计算，inception-v1在小batch_size的情况下，整个model可以直接放进IPU 304Mb-SRAM进行训练；

C.项目工作开展
1. 2/10 - 2/11 
花了一天半时间熟悉和掌握ipu相关的开发过程；
    代码：ipu_compiler_1.py --> Ipu example of regression.
         ipu_compiler_2.py --> Ipu example of classification.
2. 2/11
导入tf-slim/inception_v1 run on cpu.  代码：inception_v1_cpu_train_val.py

3. 2/12 - *
1)git push inception_v1相关代码；--Done 2/12 
2)用synthetic tensor输入到inception_v1网络中； --Done 2/12  代码：inception_v1_input.py
3)把inception_v1 compile到ipu上;  --Done 2/12  代码：inception_v1_ipu_input.py
4)git push train dataset和tf-record脚本; --Done 2/13  代码：create_labels_files.py  create_tf_record.py
5)结合train.py进行model编译; --Ongoing 2/13

D.代码go through
1.tf-slim/inception_v1.py  <--> inception_v1.png
2.inception_v1_input.py  --> synthetic tensor输入:shape=[batch_size, 224, 224, 3]
3.inception_v1_ipu_input.py  --> synthetic tensor输入:shape=[batch_size, 224, 224, 3]
4.inception_v1_ipu_train_val.py  --> ipu_compiler训练

E.Risk
1.目前不能确定在单卡ipu训练上loss是否收敛；
2.对于多卡工作还不熟悉；
