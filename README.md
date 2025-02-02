# ResUnet-a

This repository contains implementation of the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data" in TensorFlow for the detection of plot boundaries specifically.

## Docker
```sh
docker build --force-rm -f resunet.Dockerfile -t resunet .
```

```sh
docker run --rm -it \
    --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/fran/resunet_data:/resunet_data \
    -v /home/fran/ResUnet-a:/src \
    resunet bash
```


## Usage

1. Clone this repo using :
```
git clone https://github.com/Akhilesh64/ResUnet-a
``` 
2. Install the requirements using :
```
pip install -r requirements.txt
```
3. To start model training run the main.py file with following arguments :
Example folders
```
python main.py --image_size 256 --batch_size 1 --num_classes 2 --epochs 5 \
               --image_path ./images --val_image_path ./val_images --test_image_path ./test_images \
               --gt_path ./gt --val_gt_path ./val_gt --test_gt_path ./test_gt \
               --layer_norm batch --model_save_path ./ --checkpoint_mode epochs
```
```
python main.py --image_size 256 --batch_size 8 --num_classes 2 --epochs 100 \
               --image_path /resunet_data/imgs_europe/train_img \
               --val_image_path /resunet_data/imgs_europe/val_img \
               --test_image_path /resunet_data/imgs_europe/test_img \
               --gt_path /resunet_data/masks_europe/train_masks \
               --val_gt_path /resunet_data/masks_europe/val_masks \
               --test_gt_path /resunet_data/masks_europe/test_masks \
               --layer_norm batch --model_save_path ./ --checkpoint_mode epochs | tee out.log
```
<!-- ```
python main.py --image_size 256 --batch_size 1 --num_classes 2 --validation_split 0.2 --epochs 100 --image_path ./images --gt_path ./gt --layer_norm batch --model_save_path ./ --checkpoint_mode epochs
```
```
python main.py --image_size 256 --batch_size 256 --num_classes 2 --validation_split 0.2 --epochs 100 --image_path /resunet_data/images --gt_path /resunet_data/gt --layer_norm batch --model_save_path /resunet_data --checkpoint_mode epochs
``` -->
4. To produce model predictions on a directory of test images run script predict.py with the following arguments :
```
python predict.py --image_size 256 --num_classes 2 --image_path ./test --model_path ./model.h5 --output_path ./results
```
```
python predict.py --image_size 256 --num_classes 2 \
                --image_path /resunet_data/Allbands_18HYE_10m_0.00_0.20_step0.20.nc/ñuble_imgs_matched \
                --model_path /resunet_data/model_e=50.h5 \
                --output_path /resunet_data/output
```
```
python predict.py --image_size 256 --num_classes 2 \
--image_path /resunet_data/muestra_train \
--model_path /resunet_data/model_e=50.h5 \
--output_path /resunet_data/muestra_out
```
```
python predict.py --image_size 256 --num_classes 2 \
--image_path /resunet_data/Allbands_18HYE_10m_0.00_0.20_step0.20.nc/ñuble_imgs_matched \
--model_path /resunet_data/model_e=50.h5 \
--output_path /resunet_data/ñuble_imgs_matched_out
```

## Results

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Original Image** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Groundtruth** 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Predicted** 

![img1](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/images/polygon_1.png) &nbsp;&nbsp;&nbsp; ![gt1](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/gt/polygon_1.png) &nbsp;&nbsp;&nbsp; ![result1](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/results/polygon_1.png)


![img2](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/images/polygon_16.png) &nbsp;&nbsp;&nbsp; ![gt2](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/gt/polygon_16.png) &nbsp;&nbsp;&nbsp; ![result2](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/results/polygon_16.png)


![img3](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/images/polygon_46.png) &nbsp;&nbsp;&nbsp; ![gt3](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/gt/polygon_46.png) &nbsp;&nbsp;&nbsp; ![result3](https://raw.githubusercontent.com/Akhilesh64/ResUnet-a/main/results/polygon_46.png)

## Citation

The arvix version of the paper can found at the following [link](https://arxiv.org/abs/1904.00592).

If you find this repo useful please cite the original authors :
```
￼@article{DIAKOGIANNIS202094,
title = "ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data",
journal = "ISPRS Journal of Photogrammetry and Remote Sensing",
volume = "162",
pages = "94 - 114",
year = "2020",
issn = "0924-2716",
doi = "https://doi.org/10.1016/j.isprsjprs.2020.01.013",
url = "http://www.sciencedirect.com/science/article/pii/S0924271620300149",
author = "Foivos I. Diakogiannis and François Waldner and Peter Caccetta and Chen Wu",
keywords = "Convolutional neural network, Loss function, Architecture, Data augmentation, Very high spatial resolution"
}
```
