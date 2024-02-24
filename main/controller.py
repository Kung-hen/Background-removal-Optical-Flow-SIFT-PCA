# from turtle import widthnp
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from gui import Ui_Form
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

class Form_controller(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton_14.clicked.connect(self.Load_image)
        self.ui.pushButton_2.clicked.connect(self.Load_video)
        self.ui.pushButton_16.clicked.connect(self.remove_background)
        self.ui.pushButton_17.clicked.connect(self.detect)
        self.ui.pushButton_18.clicked.connect(self.track)
        self.ui.pushButton_8.clicked.connect(self.find_keypoints)
        self.ui.pushButton_9.clicked.connect(self.find_matched_keypoints)
        self.ui.pushButton_10.clicked.connect(self.find_wrap_images)
        self.ui.pushButton_12.clicked.connect(self.PCA)

    def refreshShow(self):
        height, width, channel = self.img.shape
        bytesPerLine = 3*width
        self.qImg = QImage(self.img.data, width, height,
                           bytesPerLine, QImage.Format_RGB888).rgbSwapped()    

    
    def Load_image(self):
        img,_ = QFileDialog.getOpenFileName(
            self, "Open File", "./")
        self.img = cv2.imread(img, -1)
        self.refreshShow()

    def Load_video(self):
        path, _ = QFileDialog.getOpenFileNames(
            self, "Open Video", "./", "Video Files (*.mp4 *.avi *.mov)")
        self.path = path[0]

    def remove_background(self):
        cap = cv2.VideoCapture(self.path)

        # Create background subtractor
        history = 100  # Adjust this value based on your needs
        bg_subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=1000, detectShadows=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Blur the frame
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # Apply background subtractor to get foreground mask
            mask = bg_subtractor.apply(blurred_frame)
            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Generate frame with only moving objects
            result_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Combine both
            combined = np.concatenate((frame, mask3, result_frame), axis=1)

            # Display the result and the foreground mask
            cv2.imshow("Result and Foreground Mask", combined)

            # Press 'q' to exit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def detect(self):
        cap = cv2.VideoCapture(self.path)

        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading the video.")
            return

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect corners using cv2.goodFeaturesToTrack
        corners = cv2.goodFeaturesToTrack(
            gray_frame, maxCorners=3, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        if corners is not None:
            # Convert corners to integers
            corners = np.intp(corners)
            # Draw a red cross mark at the detected point
            for i in corners:
                x, y = i.ravel()
                self.detected = (x, y)
                cv2.line(frame, (x - 20, y), (x + 20, y), (0, 0, 255), 2)
                cv2.line(frame, (x, y - 20), (x, y + 20), (0, 0, 255), 2)

        # Display the frame with the detected point
        cv2.imshow("Detected Point", frame)

        # Press any key to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.detected
    

    def track(self):
        cap = cv2.VideoCapture(self.path)

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Create a color for drawing the trajectory (Yellow in BGR)
        color = (0, 100, 255)

        # Read the first frame
        ret, old_frame = cap.read()
        if not ret:
            print("Error reading the video.")
            return

        # Convert the frame to grayscale
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        # Initialize the point to track
        p0 = np.float32([self.detected]).reshape(-1, 1, 2)

        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            # Convert the current frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow using cv2.calcOpticalFlowPyrLK
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the trajectory using cv2.line
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = np.int32(new.ravel())
                c, d = np.int32(old.ravel())
                mask = cv2.line(mask, (a, b), (c, d), color, 2)
                frame = cv2.line(frame, (a - 20, b), (a + 20, b), (0, 0, 255), 2)
                frame = cv2.line(frame, (a, b - 20), (a, b + 20), (0, 0, 255), 2)
                frame = cv2.rectangle(frame, (int(a)-10, int(b)-10), (int(a)+10, int(b)+10), (0, 255, 0), 2)

            # Overlay the trajectory on the original frame
            img = cv2.add(frame, mask)

            # Display the result
            cv2.imshow('Tracking Trajectory', img)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            # Update the previous frame and points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cap.release()
        cv2.destroyAllWindows()

    def find_keypoints(self):
        img1 = cv2.imread("Figures/Q3_Image/Shark1.jpg",0)
        img2 = cv2.imread("Figures/Q3_Image/Shark2.jpg",0)
        sift = cv2.xfeatures2d.SIFT_create(200)

        kp1 = sift.detect(img1,None)
        kp2 = sift.detect(img2,None)

        kp1 = sorted(kp1, key=lambda kp: kp.size, reverse=True)[:200]
        kp2 = sorted(kp2, key=lambda kp: kp.size, reverse=True)[:200]

        img1=cv2.drawKeypoints(img1,kp1,img1)
        img2=cv2.drawKeypoints(img2,kp2,img2)

        # cv2.imwrite(self.path + 'FeatureShark1.jpg',img1)
        # cv2.imwrite(self.path + 'FeatureShark2.jpg',img2)
        cv2.imshow('Shark1', img1)
        cv2.imshow('Shark2', img2)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def find_matched_keypoints(self):
        img1 = cv2.imread("Figures/Q3_Image/Shark1.jpg",0)
        img2 = cv2.imread("Figures/Q3_Image/Shark2.jpg",0)

        sift = cv2.xfeatures2d.SIFT_create()

        kp1 = sift.detect(img1,None)
        kp2 = sift.detect(img2,None)

        kp1 = sorted(kp1, key=lambda kp: kp.size, reverse=True)[:200]
        kp2 = sorted(kp2, key=lambda kp: kp.size, reverse=True)[:200]

        kp1, des1 = sift.compute(img1,kp1)
        kp2, des2 = sift.compute(img2,kp2)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des2,des1,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)

        img3 = cv2.drawMatchesKnn(img2,kp2,img1,kp1,matches,None,**draw_params)

        cv2.imshow("result", img3)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def find_wrap_images(self):
        img1 = cv2.imread("Figures/Q3_Image/Shark1.jpg",0)
        img2 = cv2.imread("Figures/Q3_Image/Shark2.jpg",0)
        sift = cv2.xfeatures2d.SIFT_create()

        kp1 = sift.detect(img1,None)
        kp2 = sift.detect(img2,None)

        kp1 = sorted(kp1, key=lambda kp: kp.size, reverse=True)[:200]
        kp2 = sorted(kp2, key=lambda kp: kp.size, reverse=True)[:200]

        kp1, des1 = sift.compute(img1,kp1)
        kp2, des2 = sift.compute(img2,kp2)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des2,des1,k=2)

        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append(m)

        src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        h,w = img1.shape
        result = cv2.warpPerspective(img2, M, (2*w, h))
        result[0:img1.shape[0], 0:img1.shape[1]] = img1
        cv2.imshow("result", result)
        cv2.waitKey()
        cv2.destroyAllWindows()


    def PCA(self):
    # Load RGB image
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # Convert RGB image to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Normalize gray scale image to [0, 1]
        gray_image = gray / 255.0

        # Reshape the image
        w, h = gray_image.shape
        # reshape_image = gray_image.reshape((w, h))

        # Perform PCA
        n_components = min(w, h)
        for n in range(1, n_components + 1):
            pca = PCA(n_components=n)
            transformed_image = pca.fit_transform(gray_image)
            reconstructed_image = pca.inverse_transform(transformed_image)

            # Compute Mean Squared Error (MSE)
            # mse = mean_squared_error(gray_image, reconstructed_image)
            mse = mean_squared_error(gray_image * 255.0, reconstructed_image * 255.0)
            n_mse = mse
            # n_mse = mse/255.0

            # Check if the error is less than or equal to 3.0
            if n_mse <= 3.0:
                print("Minimum components for reconstruction error <= 3.0:", n)
                break

        # Re-fit PCA with the optimal number of components
        pca = PCA(n_components=n)
        transformed_image = pca.fit_transform(gray_image)
        reconstructed_image = pca.inverse_transform(transformed_image)

        # Plot the original and reconstructed images
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Gray Scale Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f'Reconstructed Image (n={n})')
        plt.axis('off')

        plt.show()