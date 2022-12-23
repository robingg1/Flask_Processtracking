# Flask_Processtracking
Repository for our SURF project for process evaluation of construction surface

To use this repo, please install Flask framework first

Qiniu API also requires to be installed.

To download the pretrained model, please use the Link here：https://drive.google.com/drive/u/0/folders/1ZPaWKhxrvqfUqfBhRF9MdWKwQ-snOPNK

It should includes: 
   1. resnet model
   2. Pointrend model

Here are some demo pictures to show:

![输入法](https://user-images.githubusercontent.com/100852428/209273022-99f988e9-b3fb-44d2-bcb2-9a861d6370e5.png)


![surf](https://user-images.githubusercontent.com/100852428/209273008-2c58d6b3-b52b-4348-bd61-be311e97c8ce.jpg)

All the online part should use Qiniu platform for remote access, it is just a demo to show the whole pipeline.

The Pointrend model can directly being loaded using Detectron2 Pointrend Repo to make segmentation for construction surface.
