RUN:

python demo/demo.py \
--config-file configs/mapillary-vistas-65-v2/maskformer_panoptic_swin_base_transfer.yaml \
--input exs/k1.jpg exs/k2.jpg exs/k3.jpg exs/k4.jpg \
--output exs/k \
--opts MODEL.WEIGHTS output/model_final.pth \

python demo/demo.py \
--config-file configs/mapillary-vistas-65-v2/maskformer_panoptic_swin_base_transfer.yaml \
--input exs/s1.jpg exs/s2.jpg exs/s3.jpg exs/s4.jpg \
--output exs/s \
--opts MODEL.WEIGHTS output/model_final.pth \

python ./train_net.py \
  --config-file configs/mapillary-vistas-65-v2/maskformer_panoptic_swin_base_transfer.yaml \
  --num-gpus 1 \
  --resume \
  --eval-only \

python ./train_net.py \
  --config-file configs/vistas-panoptic/maskformer_panoptic_swin_base.yaml \
  --num-gpus 1 \
  --resume \
  --eval-only \

python ./train_net.py \
  --config-file configs/vistas-panoptic/maskformer_panoptic_R50_bs64_554k.yaml \
  --num-gpus 1

python ./train_net.py \
  --config-file configs/vistas-panoptic/maskformer_panoptic_swin_tiny_bs64_554k.yaml \
  --num-gpus 1

python ./train_net.py \
  --config-file configs/vistas-panoptic/maskformer_panoptic_swin_small_bs64_554k.yaml \
  --num-gpus 1

DOWNLOAD & CONVERT A MODEL:

https://github.com/facebookresearch/MaskFormer/tree/master/tools

REQUIREMENT:

pip install git+https://github.com/cocodataset/panopticapi.git

HOW TO SET LR AND BS:

https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change

[
  RandomFlip(), 
  ResizeShortestEdge(short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=800, sample_style='choice')], 
  crop: [ResizeShortestEdge(short_edge_length=[400, 500, 600], sample_style='choice'), RandomCrop(crop_type='absolute_range', crop_size=[384, 600])]

//////////////////////////////////////

python demo/demo.py \
--config-file configs/coco-panoptic/swin/maskformer_panoptic_swin_base_IN21k_384_bs64_554k.yaml \
--input exs/s1.jpg exs/s2.jpg exs/s3.jpg exs/s4.jpg \
--output exs/s/coco \
--opts MODEL.WEIGHTS model_final_4b7f49.pkl \

python demo/demo.py \
--config-file configs/coco-panoptic/swin/maskformer_panoptic_swin_base_IN21k_384_bs64_554k.yaml \
--input exs/x1.jpg exs/x2.jpg exs/x3.jpg exs/x4.jpg \
--output exs/x/coco \
--opts MODEL.WEIGHTS coco_swin_base.pkl \

python demo/demo.py \
--config-file configs/coco-panoptic/swin/maskformer_panoptic_swin_base_IN21k_384_bs64_554k.yaml \
--input exs/a1.jpg exs/a2.jpg exs/a3.jpg exs/a4.jpg \
--output exs/a \
--opts MODEL.WEIGHTS coco_swin_base.pkl \

python demo/demo.py \
--config-file configs/coco-panoptic/swin/maskformer_panoptic_swin_base_IN21k_384_bs64_554k.yaml \
--input exs/f1.png exs/f2.png exs/f3.png \
--output exs/f \
--opts MODEL.WEIGHTS model_final_4b7f49.pkl \

//////////////////////////////////////

python demo/demo.py \
--config-file configs/vistas-panoptic/maskformer_panoptic_swin_base.yaml \
--input exs/s1.jpg exs/s2.jpg exs/s3.jpg exs/s4.jpg \
--output exs/s \
--opts MODEL.WEIGHTS output_transfer/model_0244999.pth \
--opts MODEL.WEIGHTS output/model_9009999.pth \

python demo/demo.py \
--config-file configs/vistas-panoptic/maskformer_panoptic_swin_base.yaml \
--input exs/x1.jpg exs/x2.jpg exs/x3.jpg exs/x4.jpg \
--output exs/x \
--opts MODEL.WEIGHTS output/model_9009999.pth \
--opts MODEL.WEIGHTS output_transfer/model_0244999.pth \
--opts MODEL.WEIGHTS output/model_9009999.pth \

python demo/demo.py \
--config-file configs/vistas-panoptic/maskformer_panoptic_swin_base.yaml \
--input exs/f1.png exs/f2.png exs/f3.png \
--output exs/f \
--opts MODEL.WEIGHTS output_transfer/model_0244999.pth \
--opts MODEL.WEIGHTS output/model_9009999.pth \

//////////////////////////////////////

python demo/demo.py \
--config-file configs/cityscapes-19/maskformer_R101c_bs16_90k.yaml \
--input exs/s1.jpg exs/s2.jpg exs/s3.jpg exs/s4.jpg \
--output exs/s \
--opts MODEL.WEIGHTS cityscapes.pkl \

python demo/demo.py \
--config-file configs/mapillary-vistas-65/maskformer_R50_bs16_300k.yaml \
--input exs/s1.jpg exs/s2.jpg exs/s3.jpg exs/s4.jpg \
--output exs/s \
--opts MODEL.WEIGHTS vistas.pkl \

python demo/demo.py \
--config-file configs/mapillary-vistas-65/maskformer_R50_bs16_300k.yaml \
--input exs/f1.png exs/f2.png exs/f3.png \
--output exs/f \
--opts MODEL.WEIGHTS vistas.pkl \

python demo/demo.py \
--config-file configs/mapillary-vistas-65/maskformer_R50_bs16_300k.yaml \
--input exs/x1.jpg exs/x2.jpg exs/x3.jpg exs/x4.jpg \
--output exs/x \
--opts MODEL.WEIGHTS vistas.pkl \

python demo/demo.py \
--config-file configs/mapillary-vistas-65/maskformer_R50_bs16_300k.yaml \
--input exs/a1.jpg exs/a2.jpg exs/a3.jpg exs/a4.jpg \
--output exs/a \
--opts MODEL.WEIGHTS vistas.pkl 

model_1959999.pth
model_2344999.pth

2048x1024 32/16 2
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 16.906 | 65.048 | 22.176 |      116      |
| Things | 11.110 | 61.865 | 15.561 |      70       |
| Stuff  | 25.727 | 69.893 | 32.241 |      46       |

1024x512 16/16 1
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 16.887 | 67.976 | 22.202 |      116      |
| Things | 11.007 | 66.628 | 15.286 |      70       |
| Stuff  | 25.836 | 70.026 | 32.727 |      46       |

1152x576 18/16 1.125
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 17.006 | 67.668 | 22.472 |      116      |
| Things | 11.083 | 65.892 | 15.544 |      70       |
| Stuff  | 26.020 | 70.371 | 33.015 |      46       |

1280x640 20/16 1.25
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 17.321 | 67.741 | 22.795 |      116      |
| Things | 11.377 | 65.622 | 15.993 |      70       |
| Stuff  | 26.365 | 70.965 | 33.145 |      46       |

1344x672 21/16
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 17.291 | 67.080 | 22.690 |      116      |
| Things | 11.291 | 64.390 | 15.762 |      70       |
| Stuff  | 26.422 | 71.174 | 33.232 |      46       |

1408x704 22/16 1.375
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 18.003 | 67.931 | 23.512 |      116      |
| Things | 12.240 | 66.725 | 16.915 |      70       |
| Stuff  | 26.773 | 69.767 | 33.550 |      46       |

1472x736 23/16
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 17.470 | 67.280 | 22.943 |      116      |
| Things | 11.645 | 65.565 | 16.300 |      70       |
| Stuff  | 26.334 | 69.889 | 33.053 |      46       |


1536x768 24/16 1.5
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 17.766 | 67.510 | 23.359 |      116      |
| Things | 11.865 | 65.357 | 16.548 |      70       |
| Stuff  | 26.746 | 70.785 | 33.725 |      46       |

3333333333333333333333333333333333333333333333333333333333333333333333333333333

1440x720
|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 18.015 | 67.651 | 23.692 |      116      |
| Things | 12.300 | 65.604 | 17.121 |      70       |
| Stuff  | 26.713 | 70.766 | 33.692 |      46       |

555555555555555555555555555555555555555555555555555555555555555555555555555555

|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 22.553 | 70.076 | 29.433 |      116      |
| Things | 17.240 | 68.263 | 23.540 |      70       |
| Stuff  | 30.638 | 72.834 | 38.400 |      46       |


|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 21.603 | 68.954 | 28.434 |      116      |
| Things | 15.753 | 67.093 | 21.681 |      70       |
| Stuff  | 30.505 | 71.787 | 38.711 |      46       |

|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 20.997 | 69.231 | 27.408 |      116      |
| Things | 15.549 | 66.860 | 21.329 |      70       |
| Stuff  | 29.288 | 72.840 | 36.659 |      46       |

|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 22.469 | 70.270 | 29.174 |      116      |
| Things | 17.075 | 68.558 | 23.154 |      70       |
| Stuff  | 30.678 | 72.874 | 38.336 |      46       |


|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 23.352 | 61.909 | 30.324 |      109      |
| Things | 18.515 | 59.863 | 24.997 |      68       |
| Stuff  | 31.373 | 65.302 | 39.160 |      41       |

|        |   PQ   |   SQ   |   RQ   |  #categories  |
|:------:|:------:|:------:|:------:|:-------------:|
|  All   | 23.470 | 61.881 | 30.498 |      109      |
| Things | 18.649 | 59.844 | 25.189 |      68       |
| Stuff  | 31.464 | 65.260 | 39.303 |      41       |


##########################################################################

