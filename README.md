# AdaIN-Style-Transfer
Arbitrary Style Transfer is an exciting area of computer vision, where the aim is to generate an image that represents the content of one image while stylizing it with the style of another. The ability to transfer the style of a painting onto a photograph or video in real-time opens up new possibilities for artistic expression and multimedia applications. In 2017, Huang et al. proposed a novel approach called "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" that achieved state-of-the-art results in real-time style transfer. In this repository, I have implemented this approach from scratch to produce their results and provide a user-friendly notebook for users to apply the method on their images. The method described in the paper demonstrates an impressive degree of flexibility when it comes to adapting to different style images. Unlike other style transfer models that are limited to specific sets of styles, this model uses adaptive instance normalization to transfer the style of any input image onto a target content image. This means that the model is capable of reproducing the style of any given image, providing a vast range of possibilities for artistic expression. 

My implementation of ["Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf) is based on the paper by Huang et al. The encoder is a pre-trained VGG 19 network without batch normalization (BN) layers, which is used to extract the content and style features from the content image and the style image, respectively. The encoded features of the content and style image are collected and then both of these features are sent to the AdaIN layer for style transfer in the feature space.
![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/architecture.png)
AdaIN performs style transfer in the feature space by transferring feature statistics, specifically the channel-wise mean and variance. AdaIN layer takes the features produced by the encoder of both the content image and style image and simply aligns the mean and variance of the content feature to match those of the style feature, producing the target feature ***t***.

A decoder network is then trained to generate the final stylized image by inverting the AdaIN output ***t*** back to the image space generating the stylized image. The decoder mostly mirrors the encoder, with all pooling layers replaced by nearest up-sampling to reduce checkerboard effects. Reflection padding has been used to avoid border artifacts on the generated image.
 
 This model even has the ability to control the level of stylization at runtime without any additional computation or training cost.
 ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/degreesof-alpha.png)
 
This implementation is in PyTorch framework. I have provided a user-friendly interface for users to upload their content and style images, adjust the parameters, and generate stylized images in real-time.

The model has been trained for 200,000 iterations on a AWS Notebook Instance ml.p3.2xlarge(Nvidia Tesla V100 16GB) which took 16 hours approximately.

# Dependencies
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [PIL](https://pypi.org/project/Pillow/)
* [Numpy](https://numpy.org/)
* [OS](https://docs.python.org/3/library/os.html)
* [zipfile](https://docs.python.org/3/library/zipfile.html)
* [Shutil](https://docs.python.org/3/library/shutil.html#:~:text=Source%20code%3A%20Lib%2Fshutil.,see%20also%20the%20os%20module.)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [torchinfo](https://github.com/TylerYep/torchinfo)

Once you have these dependencies installed, you can clone the AdaIN Style Transfer repository from GitHub:
```bash
https://github.com/Moddy2024/AdaIN-Style-Transfer.git
```
# Key Directories and Files
* [ADAIN.ipynb](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/ADAIN.ipynb) - In this Jupyter Notebook, you can find a comprehensive walkthrough of the data pipeline for AdaIN style transfer, which includes steps for downloading the dataset, preprocessing the data, details on the various data transformations and data loader creation steps,  along with visualizations of the data after transformations and moving the preprocessed data to the GPU for model training. There's also the implementation of the AdaIN style transfer, the architecture of the model used for training and the whole training process.
* [prediction.ipynb](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/prediction.ipynb) - This notebook demonstrates how to perform style transfer on images using the pre-trained model. As this is a Adaptive Style Transfer so any style image and content image can be used.
* [results](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/results) - This directory contains the results from some of the test images that have been collected after the last epoch of training.
* [test-style](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/test-style) - This directory contains a collection of art images sourced randomly from the internet, which are intended to be used for testing and evaluation purposes.
* [test-content](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/test-content) - This directory contains a collection of content images sourced randomly from the internet, which are intended to be used for testing and evaluation purposes.
# Dataset
The Content Data which is the [COCO Dataset](https://cocodataset.org/#download) has been downloaded and extracted using wget command in the terminal. The script downloads the train2014.zip file from the official COCO website and saves it as coco.zip in the specified directory. It then extracts the contents of the zip file using the ZipFile function from the zipfile module and saves it in the content-data directory. Once the extraction is complete, the zip file is removed using the os.remove function.
```bash
!wget --no-check-certificate \
    "http://images.cocodataset.org/zips/train2014.zip" \
    -O "/home/ec2-user/SageMaker/coco.zip"

local_zip = '/home/ec2-user/SageMaker/coco.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
!mkdir /home/ec2-user/SageMaker/content-data
zip_ref.extractall('/home/ec2-user/SageMaker/content-data')
zip_ref.close()
os.remove(local_zip)
print('The number of images present in COCO dataset are:',len(os.listdir('/home/ec2-user/SageMaker/content-data/train2014')))
```
The Style Dataset has been downloaded and extracted from the Kaggle Competition [Painter by numbers](https://www.kaggle.com/competitions/painter-by-numbers/data). To ensure that the number of images in the style dataset matches the number of images in the content dataset, you can include the test dataset as well. By downloading the test dataset, you can randomly extract 3350 images from it to supplement the style dataset. This will enable you to have a balanced dataset for training your model and will also make the model robust. In order to download the dataset from Kaggle you need to extract [API Token](https://www.kaggle.com/discussions/general/371462#2060661) from the Kaggle account only then you will be able to download dataset from Kaggle to anywhere. The official instructions on how to use the [KAGGLE API](https://github.com/Kaggle/kaggle-api).
```bash
!ls -lha /home/ec2-user/SageMaker/kaggle.json
!pip install -q kaggle
!mkdir -p ~/.kaggle #Create the directory
!cp kaggle.json ~/.kaggle/
!chmod 600 /home/ec2-user/SageMaker/kaggle.json

!kaggle competitions download -f train.zip -p '/home/ec2-user/SageMaker' -o painter-by-numbers
local_zip = '/home/ec2-user/SageMaker/train.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
!mkdir /home/ec2-user/SageMaker/style-data
zip_ref.extractall('/home/ec2-user/SageMaker/style-data')
zip_ref.close()
os.remove(local_zip)
print('The number of images present in WikiArt dataset are:',len(os.listdir('/home/ec2-user/SageMaker/train')))
```

# Results
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/victoria-memorial-womanwithhat-matisse.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/lenna-picasso-seatednude.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/montreal.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/girl-brushstrokes.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/NYC.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/BANGALORE.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/NYC-scenederue.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/goldengate-starrynight.jpg)
 # Citation
```bash
@inproceedings{huang2017adain,
  title={Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization},
  author={Huang, Xun and Belongie, Serge},
  booktitle={ICCV},
  year={2017}
}
```
