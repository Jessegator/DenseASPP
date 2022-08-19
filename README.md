# DenseASPP
# DenseASPP

------

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220820052008804.png" alt="image-20220820052008804" style="zoom:70%;" />

![image-20220820042253259](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220820042253259.png)

![](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220820042159169.png)



This repository includes the pytorch implementation of [DenseASPP for Semantic Segmentation in Street Scenes](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf). Experiments are conducted on [Wiezmann Horse](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata) Dataset.

## Prepare Your Data

------

1. Download [Weizmann Horse](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata) dataset.

2. The final path structure used in my code looks like this:

$PATH_TO_DATASET/
├──── weizmann_horse_db
│    ├──── horse (327 images)
│    ├──── mask (327 images)

3. 



## Inference

Run the following command to do inference of IndexNet Matting/Deep Matting on the Adobe Image Matting dataset:

```
python main.py --phase='eval'
```

Please note that

- Save the weights under 

Here is the results of DenseASPP on Wizemann Horse dataset

| Base Model  |      Structure      | #Param. | mIoU  | Boundary IoU |                            Model                             |
| :---------: | :-----------------: | :-----: | :---: | :----------: | :----------------------------------------------------------: |
| DenseNet121 | ASPP(6, 12, 18, 24) |  10.2M  | 91.0% |    75.2%     | [Baidu Netdisk(password:2022)](https://pan.baidu.com/s/1ikRL5MeQFY2l_wZGvDmmsw) |

Note that to reproduce the results reported in table, make sure use the splits I have provided in the dataset file, which are train_list.txt and test_list.txt, consisting of names of images for training and testing, respectively.)



## Training

------

Run the following command to DenseASPP:

```
python main.py --phase='train'
```

- I randomly shuffle the dataset for training and testing: 85% for training and 15% for testing. You can do that in whatever way you like, but make sure you keep them in forms that are consistent with what I have done to train_list.txt and test_list.txt.
