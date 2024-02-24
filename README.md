
# OpenCV-Computer-Vision Practice



*Discription:*

***1.This project aims to practice various techniques in computer vision using OpenCV library to analyze and process video data. It encompasses implementing algorithms for tasks such as background subtraction, optical flow tracking, SIFT (Scale-Invariant Feature Transform), and PCA (Principal Component Analysis) for feature extraction and motion analysis.***
***2.The focus of this project focus on understanding and implementing computer vision algorithms to extract information from video.***

  * Background subtraction: This method is used to separate foreground objects from the background in a video sequence, enabling the detection of moving objects.
  * Optical flow tracking: It involves estimating the motion of objects in a video sequence by analyzing the displacement of pixels between consecutive frames. This technique is valuable for tracking object trajectories and understanding motion patterns.
  * SIFT (Scale-Invariant Feature Transform): SIFT is a feature detection algorithm used to identify distinctive key points in an image regardless of scale, rotation, or illumination changes. It's widely used in object recognition, image stitching, and 3D reconstruction tasks.
  * PCA (Principal Component Analysis): PCA is a statistical method used for dimensionality reduction by transforming the data into a lower-dimensional space while preserving the most important information. In the context of computer vision, PCA can be applied for feature extraction and pattern recognition tasks.***

![image](Figures/GUI.png)



**1.Requirements and dependencies**
  * Python 3.7 (https://www.python.org/downloads/)
  * Opencv-contrib-python (3.4.2.17)
  * Matplotlib 3.1.1
  * UI framework: pyqt5 (5.15.1)

**2.Usage:**

1. Downloads whole repository and change path into the main folder
2. Run `python start.py` .
3. Input the image 1 for feature 1 and image 2 for feature 2.
4. Run the whole code.

**3.Feature:**

1.Image Prcessing

* 1.1 Color Separation :
  
    * Extract 3 channels of the image BGR to 3 separated channels.
      
      ![image](Figures/1.1_result.png)
* 1.2 Color Transformation :
  
    * Transform image into grayscale image
    * Merge BGR separated channel images from above problem into grayscale image by average weight : (R+G+B)/3.

      ![image](Figures/1.2_result.png)
* 1.3 Color Detection
  
    * Transform [opencv.png](Figures/opencv.png) from BGR format to HSV format.
    * Generate mask by calling : cv2.inRange(hsv_img , lower_bound , upper_bound)
    * Detect Green and White color in the image by calling : cv2.bitwise_and(bgr_img , bgr_img , mask)

     ![image](Figures/1.3_result.png)
* 1.4 Blendling
  
   * Here [Dog_Strong.jpg](Figures/Dog_Strong.jpg) and [Dog_Weak.jpg](Figures/Dog_Weak.jpg) to be example

   https://github.com/Kung-hen/Image-processing-and-smooth/assets/95673520/ce2d8d34-6793-4961-8f74-fe055452e71e


    
2.Image Smoothing

* 2.1 Gaussian Blur
   * Apply gaussian filter k x k to input image1.
   * filter kernel equation = (k=2m+1)

https://github.com/Kung-hen/Image-processing-and-smooth/assets/95673520/92b5157a-ce60-4701-bbb1-2a355492ea54

* 2.2 Bilateral fliter
   * Define: Bilateral magnitude 0 ~ 10, sigmaColor = 90 and sigmaSpace = 90. 
   * Apply Bilateral filter k x k to input image1.
   * filter kernel equation = (k=2m+1)
     
https://github.com/Kung-hen/Image-processing-and-smooth/assets/95673520/ba7cc81d-efd2-4800-ba98-c87b61829303

* 2.3 Median fliter
   * Define: Median magnitude 0 ~ 10.
   * Apply Median filter k x k to input image1.
   * filter kernel equation = (k=2m+1)

https://github.com/Kung-hen/Image-processing-and-smooth/assets/95673520/89c4e7a8-0f73-4c8e-bc3d-8f68de7a81d5
