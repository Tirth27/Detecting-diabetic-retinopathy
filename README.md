# AI-For-MedicalScience
Automated DR detection system which will be provided as a service to the doctors to use it for the betterment of humanity.

## What was the purpose for selecting this project?
### Abstract
**Diabetic retinopathy** is a leading problem throughout the world and many people are losing their
vision because of this disease. The disease can get severe if it is not treated properly at its early
stages. The damage in the retinal blood vessel eventually blocks the light that passes through the
optical nerves which makes the patient with Diabetic Retinopathy blind. Therefore, in our
research we wanted to find out a way to overcome this problem and thus using the help of
**Convolutional Neural Network** (ConvNet), we wereable to detect multiple stages of severity for
Diabetic Retinopathy. There are other processes present to detect Diabetic Retinopathy and one
such process is manual screening, but this requires a skilled ophthalmologist and takes up a huge
amount of time. Thus our automatic diabetic retinopathy detection technique can be used to
**replace** such **manual processes** and theophthalmologist can spend more time taking proper care
of the patient or at least decrease the severity of this disease.

### Condition Of Diabetic Retinopathy In India

Currently, In India diabetes is a disease that affects over 65 million persons in India. 

Diabetes-related eye disease, of which retinopathy is the most important, affects nearly one out of every ten persons with diabetes, according to point prevalence estimates. Many few of them are aware of that if they have diabetes for over several years they may come across the diabetic complication.

To spread awareness among people major hospitals in India organizes the free eye checkup camps in villages where people can get their eye checkup for free. 

Those retinal images of people were collected and sent to an expert Ophthalmologist. After that, the Ophthalmologist examines those images and the summons those patients who were likely to suffer from Diabetic Retinopathy. 

This summoned patient than were informed that they are likely to suffer from Diabetic Retinopathy and should consult the expert Ophthalmologist for a proper checkup.

This whole process takes almost half a month or more and to shorten this gap we had come up with the idea which almost cut down these process into one or two days, which help the Ophthalmologist to focus more on the treatment and avoid the hectic work of identifying which patient has Diabetic Retinopathy and which doesn't.

## What was our approach?
### Why we have selected ConvNet to solve this problem? / Objective
Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. The condition is estimated to affect over 93 million people.

The need for a comprehensive and automated method of diabetic retinopathy screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. With photos of eyes as input, the goal of this project is to create a new model, ideally resulting in realistic clinical potential.

The motivations for this project are twofold:

1. Image classification has been a personal interest for years, in addition to classification on a large scale data set.

2. Time is lost between patients getting their eyes scanned (shown below), having their images analyzed by doctors, and scheduling a follow-up appointment. By processing images in real-time, EyeNet would allow people to seek & schedule treatment the same day

      ![DR Manual Screening](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/dr_scan.gif)

### From where we get dataset to train our model?
The data originates from a [2015 Kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection). However, is an atypical Kaggle dataset. In most Kaggle competitions, the data has already been cleaned, giving the data scientist very little to preprocess. With this dataset, this isn't the case.

All images are taken of different people, using different cameras, and of different sizes. Pertaining to the preprocessing section, this data is extremely noisy, and requires multiple preprocessing steps to get all images to a useable format for training a model.

The training data is comprised of 35,126 images, which are augmented during preprocessing.

### Exploratory Data Analysis
The very first item analyzed was the training labels. While there are five categories to predict against, the plot below shows the severe class imbalance in the original dataset.

![DR_vs_Frequency_table](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/DR_vs_Frequency_tableau.png)

**Confusion matrix** of **original** train **CSV**.

![trainLabels_confusion_matrix](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/trainLabels_confusion_matrix.png)

Of the original training data, 25,810 images are classified as not having retinopathy, while 9,316 are classified as having retinopathy.

Due to the class imbalance, steps taken during preprocessing in order to rectify the imbalance, and when training the model.

Furthermore, the variance between images of the eyes is extremely high. The first two rows of images show class 0 (no retinopathy); the second two rows show class 4 (proliferative retinopathy).

1. class 0 (no retinopathy)
![No_DR_white_border_1](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/No_DR_white_border_1.png)
![No_DR_white_border_2](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/No_DR_white_border_2.png)
                
2. class 4 (proliferative retinopathy)
![Proliferative_DR_white_border_1](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/Proliferative_DR_white_border_1.png)
![Proliferative_DR_white_border_2](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/Proliferative_DR_white_border_2.png)

### Different types of data preprocessing and data augmentation techniques we use to deal with major class imbalance
The preprocessing pipeline is the following:
1. [Gregwchase](https://github.com/gregwchase/dsi-capstone) approach
    - [x] **[Crop](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/1_crop_and_resize.py)** images into 1800x1800 resolution
    - [x] **[Resize](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/1_crop_and_resize.py)** images to 512x512/256x256 resolution
    - [x] **[Remove](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/2_find_black_images.py)** totally **black images** form dataset
    - [x] **[Rotate](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/3_rotate_images.py)** and **mirror**(Rotate DR images to 90°,120°,180°,270° + mirror, and only mirror non-DR images)
    - [x] **[Update](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/4_reconcile_label.py)** **CSV** so it should contain all the augmented images and there respective labels
    - [ ] **[Convert](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/5_image_to_array.py)** images to numpy array
    
2. [Ms.Sheetal Maruti Chougule/Prof.A.L.Renke](https://www.ripublication.com/irph/ijert_spl17/ijertv10n1spl_96.pdf) approach
    - [x] Image **[Denoising](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/6_Denoise_and_CLAHE.py)**
    - [x] **[CLAHE](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/6_Denoise_and_CLAHE.py)** (Contrast Limited Adaptive Histogram Equalization)
    
3. [Ben Graham](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801) approach(Only Works in python2.7)
    - [x] **[Rescale](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/Ben%20Graham/1_remove_boundary_effects.py)** the images to have the same radius (300 pixels or 500 pixels)
    - [x] Subtracted the local average color; the **[local average gets mapped to 50% gray](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/Ben%20Graham/1_remove_boundary_effects.py)**
    - [x] Clipped the images to 90% size to **[remove the boundary effects](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/src/Preprocessing%20Scripts/Train/Ben%20Graham/1_remove_boundary_effects.py)**
    
#### 1. Gregwchase approach
##### Crop images into 1800x1800 resolution
In total, the original dataset totals 35 gigabytes. All images were croped down to 1800 by 1800.

##### Resize images to 512x512/256x256 resolution
All images were scaled down to 512 by 512 and 256 by 256. Despite taking longer to train, the detail present in photos of this size is much greater then at 128 by 128.

##### Remove totally black images form dataset
Additionally, 403 images were dropped from the training set. Scikit-Image raised multiple warnings during resizing, due to these images having no color space. Because of this, any images that were completely black were removed from the training data.

##### Rotate and mirror (Rotate DR images to 90°,120°,180°,270° + mirror, and only mirror non-DR images)
All images were rotated and mirrored.Images without retinopathy were mirrored; images that had retinopathy were mirrored, and rotated 90, 120, 180, and 270 degrees.

The first images show two pairs of eyes, along with the black borders. Notice in the cropping and rotations how the majority of noise is removed.

![sample_images_unscaled](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/sample_images_unscaled.jpg)

![17_left_horizontal_white](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/17_left_horizontal_white.jpg)

After rotations and mirroring, the class imbalance is rectified, with a few thousand more images having retinopathy. In total, there are 106,386 images being processed by the neural network.

![DR_vs_frequency_balanced](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/DR_vs_frequency_balanced.png)

**Confusion matrix** of **new CSV** after image augmentation.
![trainlabel_master_v2_confusion_matrix](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/trainlabel_master_v2_confusion_matrix.png)
    
### Our neural network architecture

#### First models
Our first models used 120x120 rescaled input and I stayed with that for a decent amount of time in the beginning (first 3-4 weeks). A week or so later our first real model had an architecture that looked like this (listing the output size of each layer).

| Nr  | Name | batch | channels | width | height | filter/pool |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Input | 32 | 3 | 120 | 120 |	 
| 1 |	Cyclic slice | 128 | 3 | 120 | 120 |
| 2 |	Conv | 128 | 32 | 120 | 120 |	3//1 |
| 3 |	Conv | 128 | 16 | 120 | 120 |	3//1 |
| 4 |	Max pool | 128 | 16 | 59 | 59 | 3//2 |
| 5 |	Conv roll |	128 |	64 | 59 | 59 |	 
| 6 |	Conv | 128 | 64 |	59 | 59 | 3//1 |
| 7 |	Conv | 128 | 32 |	59 | 59 | 3//1 |
| 8 | Max pool | 128 | 32 | 29 | 29 | 3//2 |
| 9 |	Conv roll |	128 |	128 |	29 | 29 |	 
| 10 | Conv | 128 | 128 | 29 | 29 |	3//1 |
| 11 | Conv | 128 | 128 | 29 | 29 |	3//1 |
| 12 | Conv | 128 | 128 | 29 | 29 |	3//1 |
| 13 | Conv | 128 | 64 | 29 |	29 | 3//1 |
| 14 | Max pool |	128  | 64 |	14 | 14 | 3//2 |
| 15 | Conv roll | 128 | 256 | 14 |	14 |	 
| 16 | Conv | 128 | 256 | 14 | 14 |	3//1 |
| 17 | Conv | 128 | 256 | 14 | 14 |	3//1 |
| 18 | Conv | 128 | 256 | 14 | 14 |	3//1 |
| 19 | Conv | 128 | 128 | 14 | 14 |	3//1 |
| 20 | Max pool |	128 | 128 | 6 | 6 | 3//2 |
| 21 | Dropout | 128 | 128 | 6 | 6 |	 
| 22 | Maxout (2-pool) | 128 | 512 |	 	 	 
| 23 | Cyclic pool | 32 | 512 |	 	 	 
| 24 | Concat with image dim | 32 |	514 |	 	 	 
| 25 | Dropout | 32 | 514 |	 	 	 
| 26 | Maxout (2-pool) | 32 |	512 |	 	 	 
| 27 | Dropout | 32 | 512 |	 	 	 
| 28 | Softmax | 32 | 5 |

(Where `a//b` in the last column denotes pool or filter size `a x a` with stride `b x b`.)

which used the cyclic layers from the [≋ Deep Sea ≋ team](http://benanne.github.io/2015/03/17/plankton.html). As nonlinearity I used the leaky rectify function, `max(alpha*x, x)`, with alpha=0.3. Layers were almost always initialised with the SVD variant of the orthogonal initialisation (based on [Saxe et al.](https://arxiv.org/abs/1312.6120)). This gave me around 0.70 kappa. However, I quickly realised that, given the grading criteria for the different classes (think of the microaneurysms which are pretty much impossible to detect on 120x120 images), I would have to use bigger input images to get anywhere near a decent model.

Something else that I had already started testing in models somewhat, which seemed to be quite critical for decent performance, was **oversampling the smaller classes**. I.e., you make samples of certain classes more likely than others to be picked as input to your network. This resulted in more stable updates and better, quicker training in general (especially since I was using small batch sizes of 32 or 64 samples because of GPU memory restrictions).

#### Second model

First I wanted to take into account the fact that for each patient we get two retina images: the left and right eye. By **combining the dense representations of the two eyes** before the last two dense layers (one of which being a softmax layer) I could use both images to classify each image. Intuitively you can expect some pairs of labels to be more probable than others and since you always get two images per patient, this seems like a good thing to do.

This gave me the basic architecture for 512x512 rescaled input which was used pretty much until the end (except for some experiments):

| Nr | Name | batch | channels | width | height | filter/pool |
| --- | --- | --- | --- | --- | --- | --- |
| 0 |	Input | 64 | 3 | 512 | 512 |	 
| 1 |	Conv | 64 | 32 | 256 | 256 | 7//2 | 
| 2 | Max pool | 64 | 32 | 127 | 127 | 3//2 |
| 3 | Conv | 64 | 32 | 127 | 127 | 3//1 |
| 4 | Conv | 64 | 32 | 127 | 127 | 3//1 | 
| 5 | Max pool | 64 | 32 | 63 | 63 | 3//2 |
| 6 | Conv | 64 | 64 | 63 | 63 | 3//1 |
| 7 | Conv | 64 | 64 | 63 | 63 | 3//1 | 
| 8 | Max pool | 64 | 64 | 31 | 31 | 3//2 | 
| 9 | Conv | 64 | 128 | 31 | 31 | 3//1 |
| 10 | Conv | 64 | 128 | 31 | 31 | 3//1 |
| 11 | Conv | 64 | 128 | 31 | 31 | 3//1 |
| 12 | Conv | 64 | 128 | 31 | 31 | 3//1 |
| 13 | Max pool | 64 | 128 | 15 | 15 | 3//2 |
| 14 | Conv | 64 | 256 | 15 | 15 | 3//1 |
| 15 | Conv | 64 | 256 | 15 | 14 | 3//1 |
| 16 | Conv | 64 | 256 | 15 | 15 | 3//1 | 
| 17 | Conv | 64 | 256 | 15 | 15 | 3//1 |
| 18 | Max pool | 64 | 256 | 7 | 7 | 3//2 |
| 19 | Dropout | 64 | 256 | 7 | 7 |	 
| 20 | Maxout (2-pool) | 64 | 512 |	 	 	 
| 21 | Concat with image dim | 64 | 514 |	 	 	 
| 22 | Reshape (merge eyes) | 32 | 1028 |	 	 	 
| 23 | Dropout | 32 | 1028 |	 	 	 
| 24 | Maxout (2-pool) | 32 | 512 | 	 	 
| 25 | Dropout | 32 | 512 | 	 	 	 
| 26 | Dense (linear) | 32 | 10 | 	 	 
| 27 | Reshape (back to one eye) | 64 | 5 | 	 	 	 
| 28 | Apply softmax | 64 | 5 |

(Where `a//b` in the last column denotes pool or filter size `a x a` with stride `b x b`.)

Some things that had also been changed:

1. Using **higher leakiness** on the leaky rectify units, `max(alpha*x, x)`, made a big difference on performance. I started using alpha=0.5 which worked very well. In the small tests I did, using `alpha=0.3` or lower gave significantly lower scores.
2. Instead of doing the initial downscale with a factor five before processing images, I only downscaled by a factor two. It is unlikely to make a big difference but I was able to handle it computationally so there was not much reason not to.
3. The oversampling of smaller classes was now done with a **resulting uniform distribution of the classes**. But now it also switched back somewhere during the training to the original training set distribution. This was done because initially I noticed the distribution of the predicted classes to be quite different from the training set distribution. However, this is not necessarily because of the oversampling (although you would expect it to have a significant effect!) and it appeared to be mostly because of the specific kappa loss optimisation (which takes into account the distributions of the predictions and the ground truth). It is also much more prone to overfitting when training for a long time on some samples which are 10 times more likely than others.
4. Maxout worked slightly better or at least as well as normal dense layers (but it had fewer parameters).

## How we make it accessible to doctors?
In our research, to tackle the aforementioned challenges, we built a predictive model for Computer-Aided Diagnosis (CAD), leveraging eye fundus images that are widely used in present-day hospitals, given that these images can be acquired at a relatively low cost.
Additionally, based on our CAD model, we developed a novel tool for diabetic retinopathy diagnosis that takes the form of a prototype web application. The main contribution of this research stems from the novelty of our predictive model and its integration into a prototype web application.

### How the prediction pipline works? 
First start the flask app 
```
python app.py
```
1. Take the retinal image of person one per each eye
2. Upload the image to website
![upload_image_1](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/upload_image_1.jpg)
![upload_image_1_2](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/upload_image_1_2.jpg)

3. We have created a REST API which takes two images as input and return JSON response
4. The response from API is displayed into bar graph
![upload_image_2](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/upload_image_2.jpg)

5. You can also generate PDF which contain images you upload and their predictions for doctors can refer it for later use
![PDF_Generated_1](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/PDF_Generated_1.jpg)
![PDF_Generated_1_2](https://github.com/Tirth27/Detecting-diabetic-retinopathy/blob/master/images/readme/PDF_Generated_1_2.jpg)

# Note
The final model we used was in the [Model Folder](https://github.com/Tirth27/Detecting-diabetic-retinopathy/tree/master/src/Model). Also, we havetried various approach to get the good results and all the approaches are in [Miscellaneous Folder](https://github.com/Tirth27/Detecting-diabetic-retinopathy/tree/master/src/miscellaneous%20scripts).

## Credits
This project cannot be completed without you guys **Parth Purani** [@github/ParthPurani](https://github.com/ParthPurani) and **Hardik Vekariya** [@github/hv245](https://github.com/hv245). Thanks for your support :) 

## References

1. [Denoise](https://docs.opencv.org/3.3.0/d5/d69/tutorial_py_non_local_means.html) and [CLAHE](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
2. Ben Graham [1](https://github.com/btgraham/SparseConvNet) [2](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801)
3. [Gregwchase](https://github.com/gregwchase/dsi-capstone)
4. Google Research [1](https://ai.googleblog.com/2018/12/improving-effectiveness-of-diabetic.html) [2](https://about.google/stories/seeingpotential/)

