# Serialized Multi-head Multi-layer Attention

This repo contains the implementation of the paper, **"Serialized Multi-Layer Multi-Head Attention for Neural Speaker Embedding, Interspeech, 2021"**  [[Full paper](https://www.isca-speech.org/archive/interspeech_2021/zhu21c_interspeech.html)]

**Please consider to cite this paper as follows:**

```
@inproceedings{zhu21_interspeech,
  author={Hongning Zhu and Kong Aik Lee and Haizhou Li},
  title={{Serialized Multi-Layer Multi-Head Attention for Neural Speaker Embedding}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={106--110},
  doi={10.21437/Interspeech.2021-2210}
}
```



### Overview

![1630387530026](C:\Users\zhuhn\AppData\Roaming\Typora\typora-user-images\1630387530026.png)

### Implementation

We follow the pipeline and framework from [ASV-subtools](https://github.com/Snowdar/asv-subtools). Please refer to [ASV-subtools](https://github.com/Snowdar/asv-subtools) for more details.



### Here list the changed/NEW scripts from ASV-subtools:

- subtools\pytorch\libs\training\\**lr_scheduler.py**:  The training strategy we use is StepLR (Decay the learning rate of each parameter group every epoch). 

- subtools\pytorch\libs\nnet\\**pooling.py**:  Serialized Multi-head Multi-layer Attention is added. 

- subtools\pytorch\model\\**serialized-tdnn-xvector.py**:  The architecture of TDNN x-vector for serialized attention.
- Fold recipe\\**serialized-experiments**: Training and testing scripts.



### Dataset

- Training set:
  - VoxCeleb2 with augmentation
- Evaluation set:
  - VoxCeleb1-H, VoxCeleb1-E
  - SITW-dev, SITW-eval



### Training the Model

```shell
# train x-vector model with serialized attention
# training stage starts from 3. 
python subtools\recipe\serialized-experiments\runSerialized-tdnn-xvector.py --stage=3 --gpu-id=0
# train standard x-vector with statistics pooling/attentive pooling
python subtools\recipe\serialized-experiments\run-xvector.py --stage=3 --gpu-id=0
```



### Testing the Model

```shell
# This scripts is for testing on SITW-dev, SITW-eval, VoxCeleb1-H (clean) and VoxCeleb1-E (clean) 
# PLDA back-end is used
bash recipe\serialized-experiments\gather_results_sitw_voxceleb1.sh exp_name
```





