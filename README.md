# Difficulty-Guided Variant Degradation Learning for Blind Image Super-Resolution
by Jiaxu Leng, Jia Wang, Mengjingcheng Mo, Wen Lu, Ji Gan, and Xinbo Gao

This paper has been accepted by TNNLS2024! [paper link](https://ieeexplore.ieee.org/abstract/document/10709871)

## Requirement
python 3.8
torch 2.1.0
torchvision 0.16
scipy 1.10.1

## Test
```git clone https://github.com/JiaWang0704/DDL-BSR.git
cd DDL-BSR
python test.py -opt=options/test/test_setting2_stage3_x4.yml
```

## Train
```git clone https://github.com/JiaWang0704/DDL-BSR.git
cd DDL-BSR
python train.py -opt=options/train/train_setting2_stage3_x4.yml
```


## Citation
If you use this code or use our pre-trained weights for your research, please cite our papers:
```@article{leng2024difficulty,
  title={Difficulty-Guided Variant Degradation Learning for Blind Image Super-Resolution},
  author={Leng, Jiaxu and Wang, Jia and Mo, Mengjingcheng and Gan, Ji and Lu, Wen and Gao, Xinbo},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```


