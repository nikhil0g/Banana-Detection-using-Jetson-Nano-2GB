# Banana-Detection-using-Jetson-Nano-2GB
## AIM AND OBJECTIVES
## AIM
#### To create a classification system which detects the quality of a Banana and specifies whether the given Banana is rotten or not.

## OBJECTIVES

 • The main objective of the project is to do quality check of a given Banana within a short period of time.
 
 • Using appropriate datasets for recognizing and interpreting data using machine learning.
 
 • To show on the optical viewfinder of the camera module whether a Banana is rotten or not.

## ABSTRACT

• A Banana can be classified based on it’s quality and then specified whether it is rotten or not on the live feed of the system’s camera.
    
• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one   another. Machine Learning provides various techniques through which various objects can be detected.
    
• One such technique is to use Resnet Net-50 model with Keras, which generates a small size trained model and makes ML integration easier.
    
• Huge post–harvest losses are caused because of poor handling while shipping bananas, because of which bananas sometimes gets crushed and hence develops black     spots which further increases the process of rot. 


## INTRODUCTION

• This project is based on Banana detection model. We are going to implement this project with Machine Learning.

• This project can also be used to gather information about the quality of Bananas like whether they are rotten or not.
    
• Banana quality can be further classified as raw, slightly raw, ripe, over ripe and rotten.
    
• Based on which side of Banana appears on the camera’s viewfinder the Banana may appear ripe or rotten hence using multiple cameras or rotation of the Banana is   recommended.
    
• Today, each and every system is being automated. There is much less human intervention as it is both time consuming and non-economical. For the defect           identification purposes, automated systems are used. The image processing has been widely used for identification, classification and quality evaluation.
    
• Here a sorting process is introduced where the image of the Banana is captured and analysed using image processing techniques and the defected Banana is discarded by this process. The main aim is to do the quality check of the Bananas within a short span of time.
   
• Due to cost and inaccurate process, sorting tons of quality Bananas to produce food products made from Bananas is an another problem that is faced by most of the agricultural industries.
    
• Neural networks and machine learning have been used for these tasks and have obtained good results.
    
• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Banana detection as well.


## LITERATURE REVIEW

• Fresh Bananas are an important part of a healthy diet. They contain essential vitamins, minerals, fiber and other nutrients that are essential for good health.   In fact, research has shown that a healthy diet rich in Bananas may reduce the risk of cancer and other chronic diseases.
    
 • Bananas are rich in fiber which is very essential for the smooth movement of the digestive system. There are some Bananas that give body energy as they contain   carbohydrates which are the main source of energy. Carbohydrates in Bananas are mainly sugar which actually breaks down easily and make a quick source of        energy.
    
 • Antioxidants are essential for human health. These compounds mop up free radicals in the body that can damage the body’s cells and lead to diseases, such as cancers.Researchers believe that the Bananas contain Trusted Source tryptophan, an amino acid that may help preserve memory, boost a person’s ability to learn and remember things, and regulate mood.
    
 • As the idiom goes one rotten apple spoils the barrel so does any Banana can also spoil every Banana surrounding it hence finding the rotten ones as fast as possible becomes absolutely necessary.
    
 • Most fresh Bananas are picked before they are ripe. This allows them time to fully ripen during transportation. It also gives them less time to develop a full range of vitamins, minerals and natural antioxidants. In the US, Bananas and vegetables may spend anywhere from 3 days to several weeks in transit before arriving at a distribution center.
    
 • During transportation, fresh produce is generally stored in a chilled, controlled atmosphere and treated with chemicals to prevent spoiling. Once they reach the supermarket, Bananas and vegetables may spend an additional 1–3 days on display. They’re then stored in people’s homes for up to 7 days before being eaten.
    
 • Shortly after harvesting, fresh Bananas start to lose moisture, have a greater risk of spoiling and drop in nutrient value. The vitamin C in fresh Bananas begins to decline immediately after harvesting and continues to do so during storage. 
    
 • Food makes the largest source of waste that goes in muncipal landfills and combusted for energy recovery. Many a times the Bananas from vendors or shopkeepers go to waste because of the Bananas turning rotten because of one Banana that was not taken out early.


## PROPOSED SYSTEM

1] Study basics of machine learning and image recognition.

2] Start with implementation

        A. Front-end development
        B. Back-end development
        
3] Testing, analyzing and improvising the model. An application using python IDLE and its machine learning libraries will be using machine learning to identify whether a given Banana is rotten or not.

4] use datasets to interpret the object and suggest whether a given Banana on the camera’s viewfinder is rotten or not.


## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.
    
• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
    
• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.


## Jetson Nano 2GB







https://user-images.githubusercontent.com/89011801/151480525-4615fe18-cbf2-4dd2-a954-2efbe6627558.mp4




## Installation

#### Initial Configuration

```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*

```
#### Create Swap 
```bash
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1	swap	swap	defaults	0 0
```
#### Cuda env in bashrc
```bash
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

```
#### Update & Upgrade
```bash
sudo apt-get update
sudo apt-get upgrade
```
#### Install some required Packages
```bash
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
```
#### Install Torch
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
#### Install Torchvision
```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
#### Clone Yolov5 
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
```
#### Download weights and Test Yolov5 Installation on USB webcam
```bash
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0
```
## Banana Dataset Training

### We used Google Colab And Roboflow

#### train your model on colab and download the weights and pass them into yolov5 folder.


## Running Helmet Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Demo





https://user-images.githubusercontent.com/89011801/153808366-50556f24-f92a-4ccc-b07d-b59bb0537332.mp4





## ADVANTAGES
    
• Banana detection system will be of great help in minimizing the damage that one rotten Banana can do to the entire barrel of Bananas.
   
• Banana detection system shows whether the Banana in viewfinder of camera module is rotten or not with good accuracy.
    
• Once a rotten Banana is detected on the camera module the Banana detection model can they convey it to the worker in the shop or plant or if needed this process can be completely automated to make machines do the work of sorting Banana.
  
• When completely automated no user input is required and therefore works with absolute efficiency and speed.
  
• It can work around the clock and therefore becomes more cost efficient.


## APPLICATION
   
• Detects a Banana and then checks whether the Banana is rotten or not in a given image frame or viewfinder using a camera module.

• Can be used in shops, malls, plants, storage facilities, godowns and at any place where Bananas are stored and needs sorting.

• Can be used as a refrence for other ai models based on Banana Detection.

## FUTURE SCOPE

• As we know technology is marching towards automation so this project is one of the step towards automation.
  
• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

• Banana detection will become a necessity in the future due to rise in population and hence our model will be of great help to tackle the situation of starvation and malnutrition due to food shortage in an efficient way.

## CONCLUSION

• In this project  our model is trying to detect a Banana and then showing it live as to whether Banana is rotten or not. The Banana detection system is used to automate the segregation of Bananas from rotten ones in an efficient way.
   
• The model tries to protect the shopkeepers and plant owners from incurring any losses.

•  The model is efficient and highly accurate and hence reduces the workforce required.
  
  
## Reference

#### 1] Roboflow:- https://roboflow.com/

#### 2] Google images

## ARTICLES

#### https://timesofindia.indiatimes.com/city/trichy/half-of-post-harvest-banana-loss-caused-by-poor-storage-system/articleshow/74261788.cms

#### https://www.vahrehvah.com/indianfood/importance-of-fruits-for-healthy-life

#### https://www.canr.msu.edu/news/preserving_vitamins_in_fruits_and_vegetables






