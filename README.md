# 700_deep_set
This is the repo for CIS700 final project!

If there are any issue with the code, please let us know! Thanks a lot!

Environment Setup: pip install -r requirements.txt

Run code for different task:

Copy_v0, DuplicatedInput_v0 and reverse input sequence use the same model configurations for testing, we change the number layers in the permutation equivariant stack(If you want to test different configuration, please change the code in ./model/Deepsets.py, and then the permutation equivariant stack is from line 58 to 69.)
The training curve will be generated in task_0_fig, task_1_fig and task_3_fig folder.

The generating of the data need about 2 to 4 mins, please wait. Thanks for the patience.

Copy_v0(Weiheng Chai):

1.First step is generating dataset:

cd data

python generate_data_task_0.py

cd ..

2.Second step is training the model:

python train_task_0.py

3.The third step is testing on different datasets:

python test_task_0.py



DuplicatedInput_v0(Feng Wang):

1.First step is generating dataset:

cd data

python generate_data_task_1.py

cd ..

2.Second step is training the model:

python train_task_1.py

3.The third step is testing on different datasets:

python test_task_1.py




Repeat Copy(Minmin Yang):

For different configurations, please change the number of layers in permutation equivariant stack, as we mentioned in the report, and the input dim can be changed in line 39 of the code, change the d_dim number.

1.First step is generating dataset:

cd data

python generate_data_task_2.py

cd ..

2.The sencond step will train and test model:

python train_test_task_2.py



Reverse Input Sequence(Siling Yin):


1.First step is generating dataset:

cd data

python generate_data_task_3.py

cd ..

2.Second step is training the model:

python train_task_3.py

3.The third step is testing on different datasets:

python test_task_3.py
