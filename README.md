# DeepSTD: Mining Spatio-temporal Disturbances of Multiple Context Factors for Citywide Traffic Flow Prediction

<p align="center">
  <img width="500" height="400" src=./figure/DeepSTD.png>
</p>

This is the implementation of DeepSTD in the following paper: \
Chuanpan Zheng, Xiaoliang Fan, Chengwu Wen, Longbiao Chen, Cheng Wang, and Jonathan Li. "[DeepSTD: Mining Spatio-temporal Disturbances of Multiple Context Factors for Citywide Traffic Flow Prediction](https://ieeexplore.ieee.org/abstract/document/8793226)", published in *IEEE Transactions on Intelligent Transportation Systems* *(T-ITS)*.

## Data

The Chengdu dataset with 500m*500m grid size and 15-minute time interval is provided in the './data' folder.

## Requirements

Python 3.7.10, tensorflow 1.14.0, numpy 1.16.4, pandas 0.24.2

## Results

We provide a pre-trained model, which achieve the following performance:

|    Chengdu     |  Workday   |   Weekend    |  All days  | 
| -------------- | ---------- | ------------ | --------   | 
| DeepSTD        | 10.36      | 10.78        | 10.48      | 

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@article{ DeepSTD:TITS,
  author   = "Chuanpan Zheng and Xiaoliang Fan and Chenglu Wen and Longbiao Chen and Cheng Wang and Jonathan Li"
  title    = "DeepSTD: Mining Spatio-temporal Disturbances of Multiple Context Factors for Citywide Traffic Flow Prediction",
  journal  = "IEEE Transactions on Intelligent Transportation Systems",
  volume   = "21",
  number   = "9",
  pages    = "3744--3755",
  year     = "2020"
}
```
