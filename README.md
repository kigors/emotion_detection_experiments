# Emotion Detection Experiments

## Introduction.
I analyzed different training strategies of convolutional neural networks (CNN) for emotion detection based on images (face photos). I experimented with Inception v3 and BiT-M r50x1 pre-trained networks (trained on ImageNet), compared fine-tuning and feature extraction means of transfer learning.

*(Note: notebooks are in Russian but code comments are in English)*

## Used libraries.
1. Python 3.8.8
2. python packages:
    - tensorflow 2.4.1
    - tensorflow_hub 0.11
    - cv2 4.5
    - numpy
    - pandas
    - pillow 8.1.2
    - matplotlib
    - tqdm
    - livelossplot 0.5.4
3. Jupyter Notebook or JupyterLab of your choice.

## Getting started
To start working with repo, do the following preparations:

1. Download dataset files from [Kaggle competition](https://www.kaggle.com/c/skillbox-computer-vision-project/data) and unzip in `./dataset` folder. Eventually you will have this folder structure:
```
    ./dataset
      /train
        /anger
        /contempt
        /disgust
        etc.
      /test_kaggle
        <unstructured images>
```
2. Download opencv model files for face detector:
    - [model](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel) rename to `opencv_face_detector.caffemodel`
    - [config](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt)
    - place both files to `./face_detection_data` folder.
3. If you want to play with emotion detection in real-time from web-camera, you'll need to download any truetype font you like, for example from [Google Fonts](https://fonts.google.com/). You will have to edit *video_cam.py* file and change the line `BoxDrawer(font_ttf=' ... ')` with this filename (under `__name__ == '__main__'` section).

## Data analysis
My analysis starts with [EDA_and_Train_model.ipynb](https://github.com/kigors/emotion_detection_experiments/blob/master/EDA_and_Train_model.ipynb) file. Look through dataset images, get some statistics, prepare some custom filters for extra augmentation, split data into train and validation datasets.

## Experiments with models
In the next notebook [Model_training_experiments.ipynb](https://github.com/kigors/emotion_detection_experiments/blob/master/Model_training_experiments.ipynb) I build models using pre-trained CNN models from TensorFlow Hub without classifier head : 
- [Inception V3](https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4) ([ "Rethinking the Inception Architecture for Computer Vision", 2015](https://arxiv.org/abs/1512.00567))
- [BiT-M r50x1](https://tfhub.dev/google/bit/m-r50x1/1) ([Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370))

For transfer learning I used feature extraction on Inception model and fine-tuning on BiT model. At the end I trained Inception model with fine-tuning. Feature extraction learning (freeze pre-trained model and learning new head only) showed significantly less categorical accuracy than fine-tuning (training both pre-trained model and new head simultaneously). Fine-tuning was done following BiT-HyperRule: [BigTransfer (BiT): State-of-the-art transfer learning for computer vision](https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html).

I prepared 2 datasets for training:
- original images
- images of faces (cropping faces from original images using openCV face detector)

Both approaches led to quite similar results. Nevertheless, I would recommend learning on original images and it gives more room for image augmentation (shift, rotate, zoom, etc.) without cropping face too much (in other words, keeping eyes and mouth on the image). Therefore I can make you classifier more reliable.

## Experiments with valence-arousal representation.

I the next notebook [Valence-arousal model.ipynb](https://github.com/kigors/emotion_detection_experiments/blob/master/Valence-arousal%20model.ipynb) I made a regression model to predict valence and arousal values of a face on a photo.

Emotions can be defined in a two dimensional space: valence and arousal.
> The valence dimension (emotional pleasantness) describes the positivity or negativity of an emotion and ranges from unpleasant feelings to a pleasant feeling (sense of happiness). The arousal dimension (physiological activation) denotes the level of excitement that the emotion depicts, and it ranges from Sleepiness or Boredom to high Excitement. 
> [Article: Kriging Predictor for Facial Emotion Recognition Using Numerical Proximities of Human Emotions](https://informatica.vu.lt/journal/INFORMATICA/article/1182/text)

The main issue here is that given dataset wasn't labeled with valence-arousal values, only classes. Thus I had to assign it based on class (**each class got fixed valence-arousal values, aka. emotion center on the plane**). I used results of the article: [Multidimensional Emotion Recognition Based on Semantic Analysis of Biomedical EEG Signal for Knowledge Discovery in Psychological Healthcare](https://www.mdpi.com/2076-3417/11/3/1338/htm) to map given 9 emotions on valence-arousal plane.

Worth saying, the best approach is to label images with valence-arousal values by a group of trained people (shouldn't be psychologists but instructed to understand the difference between these dimensions). It's quite subjective process and requires some sort and averaging/processing. I didn't have such data.

I trained Bit-M r50x1 model in 3 different ways:

1. Two headed model: regular classifier (Dense(9)) + regressor (Dense(2)). Loss was mixed 80/20 between these two heads so the whole model was dominated by classifier and regressor had to follow it. It was done to avoid MSE loss ruining pre-trained weights on BiT model (it was my first assumption).
2. Regressor model - single head with 2 neurons: one is for valence value, and another is for arousal value. 
3. Same model as in case 2 but with L2-regularization on the head and standardization of target valence and arousal values.

Eventually, all three approaches came up with very similar results when I try to predict emotion class based on the distance between predicted valence-arousal values and emotion centers. But what is interesting: on many photos I tend to agree with V-A values predictions. In other words, I believe that predicted valence value is quite right in measuring positivity or negativity of emotion and quite often I agree with V-A values but do not agree with the label, which was assigned to the photo. Anyway, I see this method of predicting V-A values being quite useful when it comes to how positive or negative emotion is even in real-time (both models inference time was about 14ms or 70fps running on GPU), or you can log valence im time and correlate with whatever you study (Kaggle competition's legend is to study test group emotions while watching new TV shows or movies).

## Real-time emotion labeling from web-cam.

I wrote a simple enough script which capture a frame from your USB web-camera, detect faces on it, predict emotion and visualize in similar way as given on Kaggle page. ![example](https://miro.medium.com/max/1400/1*rSOC2rIKZ3NSkE3j1MetdQ.png)

Related files are: `video_cam.py`, `emotion_classifier.py` + your ttf font.

## Trained models
- Inception V3 [checkpoint](https://drive.google.com/file/d/1ItfDKQmCGKxA-b2Dhknil9_AW6Mp0xMv/view?usp=sharing)
- BiT-M r50x1 [checkpoint](https://drive.google.com/file/d/1jiknkqJmwvZqeVplntZyfj9nxvYYJHzV/view?usp=sharing)
