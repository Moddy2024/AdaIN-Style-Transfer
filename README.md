# AdaIN-Style-Transfer
Arbitrary Style Transfer is an exciting area of computer vision, where the aim is to generate an image that represents the content of one image while stylizing it with the style of another. The ability to transfer the style of a painting onto a photograph or video in real-time opens up new possibilities for artistic expression and multimedia applications. In 2017, Huang et al. proposed a novel approach called "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" that achieved state-of-the-art results in real-time style transfer. In this repository, I have implemented this approach from scratch to reproduce their results and provide a user-friendly interface for users to apply the method on their images.

My implementation of ["Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf) is based on the paper by Huang et al. The encoder is a pre-trained VGG 19 network without batch normalization (BN) layers, which is used to extract the content and style features from the content image and the style image, respectively. Then features collected from both of these images are then combined using an adaptive instance normalization (AdaIN) layer.

AdaIN performs style transfer in the feature space by transferring feature statistics, specifically the channel-wise mean and variance. AdaIN layer takes the features produced by the encoder of both the content image and style image and simply aligns the mean and variance of the content feature to those of the style feature, producing the target feature t.
 
 This model even has the ability to control the level of stylization by adjusting the number of style layers used in the AdaIN layer.
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

Once you have these dependencies installed, you can clone the Bird Classification repository from GitHub:
```bash
https://github.com/Moddy2024/AdaIN-Style-Transfer.git
```
# Key Directories and Files
* [ADAIN.ipynb](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/ADAIN.ipynb) - This file shows how the dataset has been downloaded, how the data looks like, the transformations, data augmentations, architecture of the model the training.
* [prediction.ipynb](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/prediction.ipynb) - This file loads the trained model file and shows how to do predictions on single images.
* [results](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/results) - This directory contains the results from some of the test images.
* [test-style](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/test-style) - This directory contains test art images collected randomly from the internet.
* [test-content](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/test-content) - This directory contains test content images collected randomly from the internet.
# Dataset
The Content Data which is the COCO Dataset has been downloaded and extracted using wget command in the terminal. The script downloads the train2014.zip file from the official COCO website and saves it as coco.zip in the specified directory. It then extracts the contents of the zip file using the ZipFile function from the zipfile module and saves it in the content-data directory. Once the extraction is complete, the zip file is removed using the os.remove function.
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
The Style Dataset which is the WIKIArt Dataset has been downloaded and extracted from the Kaggle Competition painters by number
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
