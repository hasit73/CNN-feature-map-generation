# CNN-Feature-Map-Generation

### Backend : Tensorflow
### Library required:

- opencv = '4.5.4-dev'
- matplotlib='3.3.4'
- tensorflow='2.6.0'
- tqdm='4.62.3'


# Quick Overview about structure

#### 1) map_generation.py

- Loading model and user configurations
- generate feature map for input image and save feature map


#### 2) map_gen.ipynb

- this file contains just a sample piece of code
- feature map generation code 


# How to use 

1) clone this directory

2) use following command to run feature map generation on your input image

  ```
  python main.py -m <model_name> -d <depth_value> -i <source_image>
  ```

  Ex: 
  ```
  python main.py -m resnet50 -d 5 -i sample.png
  ```


### Results

feature map samples


![vgg_conv2_block1_preact_bn_FusedBatchNormV3_0](https://user-images.githubusercontent.com/69752829/142006274-b3b868c7-f7a7-4b85-9401-3dbccda56a16.png)

![vgg_conv2_block1_preact_relu_Relu_0](https://user-images.githubusercontent.com/69752829/142006325-9134aeac-db71-4dab-bfec-396a3a00bd1b.png)






