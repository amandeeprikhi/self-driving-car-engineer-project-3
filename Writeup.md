# **Traffic Sign Recognition** 

## Writeup

### Files submitted:
* **Traffic_Sign_Classifier.html**
* **Traffic_Sign_Classifier.ipynb**
* **Writeup.md**
* **Writeup.pdf**

---

# Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed for various datasets.

![Training Data](https://lh3.googleusercontent.com/ECbixt5RSvCwQCMWFQH_dNtGaSXZrgTnXWYY9P4MaCajy7Ylkh7Py_CGHYmrwZVDI18fX3gVuLhba4LdUb4oc2Bla3A_3JsrxDseanbWB89JjIMomSs7uhy7SKK6r8VCBVk9LTPqN_4siiKLxcYi3x8Z3IF-ZwHp8yHLhdwFVv6_wsAvwS7NOM5fZIy7E3tV1YP1-21tVCb1oeT3YTmPzvjX8S0UQSkyybEYBNPdLMO_piLyJuzvm8glm-AQv6i4qLB2LQ9yZYrw7yrWjVyGumtgkZ4OaA-e1WTZVfEgmX_fF5JL19BYTSsbMdCrej9S7olWawcGxPNGcdjJ5FQHcOra_mJDyOcCk0TGnUGFKaoVkAsTEOjuRsr4ltGeC-naX5GoKcfttGJhJQ3mUJ4u76g7OccoeGOVRJWalULv9BIkio1rWzzIgBVR3A6W8sJRtnmzvWDzo2M0_jtXkUvMGlrwimvdWetJkRJIDZJdhI1iSg3fq2L9Nu02Z8H0J4PjliNjRqtfYLzKqZhAkrNgn2Hcm3Y4_OCXfOGHLhl-P-Vok_s5MaVCTWFE0gL2nBCQnMyJZmXCzeSYaALNmw43oz80c60NVDHdWgMpEoyz6f88ZXcKMpHUw0fVGJKDj39s2q92e4mTUSyRugN7qF1GXMaTAzTtywU=w378-h263-no)
![Validation Data](https://lh3.googleusercontent.com/1F2hjRsuxVW7F44ud6ekVSP-9qjZJMnhJoS6h63BdMIzWqjX-28PaJ1MPqrWF9U_-4rj3gcdDuWPSDDOuAPP362WlDWf-fCgiOeiNKZG8RuS8gDlYwuSgESM9Ingp4WBMVNaho6nqyKhxaHUxei0ubhr8fprnoFftvGPOb9piHhEAHsavEjUHKcQn0ys91CA6s8BzXhmW7kYVDk26oazQP6VtNEZiu0zw1pgSFD5U7OMQLNOves1G7wZbBtyN0rhnSp-KsP7jUp2aQdBLFtSTmc0rC4ng4qCwLC7cV4liMjpgxdC_n5KbGzMTuGfDXQanA650U50LBhe8uXzqZ9HQq4u1YGXZVDaIqWuE9y5bLomhAK17ik66H4a4wRTQEj5hFGa3GmdSM-IgduNuqD7mvFBVTdn7iUZwnNqJu1v2fGRJbf1RS5C6ptqM9R0LrJ6bx3tWNX2y5CmbCzbBBYfO0rZ-zSX4XAAZkkf9hXM3UIqx165910MISWsCF84MFYBeKS1vAblJxnf0N_L_4G-Kc2b70M9LZnm6rwLuMjpw7G4F_WtVGSXSWLtBJ-zJQFG2UxeHQtq_aUUZszSWekWh57U4ezDHoDS9Kj-fy5nnxIXS6-LraRXKbVfU1v7xSlUSAfZ1PSb-D9LkiLf6sd9h6dsrnI1Kgo=w373-h263-no)
![Testing Data](https://lh3.googleusercontent.com/o1AVgoChVyqZPKLG8WRTVVUAzQDEc3HCcdro6SE8cnl2wZvO-LZO_u7PTGCBepJOqizI0uI8UpDINULcGtQXuaWQCdaRP7GbQGrNkR62lWsyu_zTpkFMlsDHmj2QxTTNd9zdMRQzTfk3KnjkJS1r9WM8CjAcrCXnfQRxy-ZKiCLTE2YqMJ0mWZv41wXC9XfkPv8B-3cBZWWQRxoAUBft-Ki0vSOQBTrhlypZg6ovW3engXx2SuhfKgdCuiQ5MeNe9iloysfmv8JUpuO_zvLhF_n7Vc6jWi49wWdADcoNprEQFouB64CyvOIXRJZEXBQtTYa87U40BOHFxhAaTA8Y6bsC3Wh02MRi5NXKtqIRJKuiwtrl1n6INXlxjrYZ_puUo3UcevNbaAyG9ZIN2AhI1KT0ArDQFndeMK_cufa_8jt5jCPEHhveFl8yY2oRQ2i2IGdkTF12BSW52sy7x3mNppGKJqWM5EuNBuYAoFDxYkgbkiYLuTNugMuizYruacn5bWGThlpz9fj3Kdue7LAJ1Ujsl3t924TocMQdr2NH0Ljf8KccwxntIYPJSP4D7hrQHP1sZ2gYcmyBct8aedPDXO6Gjc2wIOmLwDNe_--crDcUahYcv4xLXc_Bd9pffNYkjimh0c85x7_TaSGJuqBVyIXnF7pCA5U=w373-h263-no)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a part of pre-procesing the image dataset, I normalized the images in the method **pre_process(img)**.
It takes individual images from the whole dataset and normalizes it in order to attain **0 mean** in the dataset.
The function is as follows:
```
def pre_process(img):
  return (img - 128.)/128.
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   						    	| 
| Convolution        	| 1x1 stride, valid padding, outputs 28x28x6 	    |
| RELU					| Activation function  							    |
| Dropout   	      	| Helps in reducing the over-fitting		    	|
| Max Pooling   	    | Input = 28x28x6,same padding output = 14x14x6 	|
| Dropout       		| Helps in reducing the over-fitting		    	|
| Convolution   		| Input = 14x14x6, vlid paading, output = 10x10x16  |
| RELU          	    | Activation function 						    	|
| Dropout				| Helps in reducing the over-fitting				|
| Max Pooling         	| Input = 10x10x16,same padding, output = 5x5x16   	| 
| Dropout        	    | Helps in reducing the over-fitting 	            |
| Flatten				| Input = 5x5x16, output = 400  					|
| Fully Connected Layer | Input = 400, output = 120		    	            |
| RELU   	            | Activation function 	                            |
| Dropout       		| Helps in reducing the over-fitting		    	|
| Fully Connected Layer | Input = 120, output = 84                          |
| RELU         		    | Activation function   						    | 
| Dropout        	    | Helps in reducing the over-fitting 	            |
| Fully Connected Layer	| Input = 84, output = 43  							|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an **Adam Optimizer**
Furthermore, various other hyper-parameters are as follows:
* **Learning Rate** = 0.001
* **Batch Size** = 128
* **Number of Epochs** = 50

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **1.000**
* validation set accuracy of **0.954**
* test set accuracy of **0.949**

The following steps were taken while implementing the solution:
* First, I used the **LeNet** architecture that was taught to me in the previous lessons of the nanodegree program.
    * It had **10 EPOCHS** and a **batch size** of **128**.
    * I was able to attain an accuracy of around **85%** after training was completed.
* Then, I moved onto feeding in the pre_processed image data that had 0 mean value.
    * The accuracy increased but not by much.
    * But, I was seeing an upward trend in the validation accuracy. So, I moved onto the next step.
* I increased the number of epochs to **50**.
    * The accuracy did increase but it plateaued at around **90%**.
* Then, I increased the number of EPOCHS even more.
    * Number of EPOCHS = 150
    * The accuracy remained same i.e., 90%.
    * The model resulted in overfitting and thus the validtion accuracy didn't increase.
* After this, I tried lowering the learning rate so as to achieve higher accuracy.
    * But, the results didn't vary much.
* Finally, I introduced dropouts into the model in order to prevent the over-fitting.
    * I first introduced dropout after the first **convolutional layer**.
    * Kkeeping the numebr of EPOCHS as same the accuracy for the validation set increased as the training accuracy took longer to plateau this time.
* So, I introduced more dropouts to the model.
    * Appartenly, the model achieves the accuracy of **0.93** for the validation set at around 30 EPOCHS.
    * But, I wanted to attain maximum accuracy for the validation set hence, I've kept the EPOCHS as 50 and BATCH SIZE as 128 for the final model architecture.
* The final model was quite different from LeNet architecture also in terms of input and output.
    * The LeNet architecture had input images with 1 channel while traffic sign images are 3 channel.
    * Also, the output of the LeNet architecture had 9 nodes, whereas the current model has 43 output nodes.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Right of way](https://lh3.googleusercontent.com/KX2pdgbO6KtqGKH3cpGJwvIms83yfFpEr7iEAxMj5miwjddD5KgWWKJZEKNhhcTnOZqRXw-K0v61P8eADxSDXz1sxzS00GGRuVe0AAuOylBgTrivBDnHMf4DR0mal-Yfr1rDryAGkdpqn4W-vlYNHlAyW-fHgF4CjGkr-uG4fglAzyaRFXaoM-X1wPjiSzEJdpM7zqCidywaG94h4TreOMOVBAaSNJlf5UD8d2hRwifQdPiKQTs_hRlBjlIaPHyCClUfQfNH8znRFFx974CoQjSu7tdVgJvwvgiuqIdIgObil6ldYCVli89hfAFnSrF8OmMw0siigYx1m6zVB6dHJR19QqFEdX07eyzwB6af6C0v9DqXFCAPHknrlIq_FOK38mBTOjxBQ_iCsVtFbbsW2cqR6gDV4Gry66YAHgTG8rfHXLGhYI1ZvAZCRBMP4pkmb7Uqj4waanVrawge_UzgFKGoshbHsWEoGATvZlDlHabiM9AxIX_R_0NN55kFLOtlM5xfWPNalqVUd66pvO_vXo4IUpz5MOAxJyUV1d7DU0FIE0pxBQiAlX2SrAVlUFWNddN8lE3kJWwlIyl3WZmkxEW7n8y3DZw7gYbXZjY_ZYVo07gQdVhDeZcOUL_FtqXjSY-NjVs1t8BbymOEK5AXhg9J5rzEezU=w633-h632-no) 
This above image might be difficult to classify because the traffic sign represented above has a triangular boundary which makes it similar to other traffic signs of the same shape. Such as the **General Caution** sign below.
The image has a higher resolution and size than what my model accpets, this image needs to be re-shaped to be fit into the classification model.
![Priority Road](https://lh3.googleusercontent.com/7pC0V8EtpkMuyn8KVgSeyOimwVpTTO9wSdrowZWXKVh-JVilFrO627TF4P6pvJIa40BgawM0mlOMf7M27kJzbh5zp0UZr2jK3h7Us_gIqOXjxpMyUOI8FES0No2S_mM1q7k78Dfrzbqex_0ouI3361uCPWJ6JD2VNxcnq8NjcJmPCiCD0DhNRxy2tWDwEoT7KwKY69sCd0f80FIYJSm3Eh9uFtp_drOTiGTqAqj_KFz9X6AJDt2N1SdgYylVo5ghRo6R5wpj-7BD8OoJOfQSbC-pVuvvvZHWuyYNPn5xQBij3gAoNNiP5ViT0P_I0HDHVpZbu3xDZGG_pD-1Mx8cz4f8ptYarPNBVGDXXcfcBhi6lwra09BYiKUS8D_FGzr3z69IQQRNN9BmXrdj0ZoZLKZbimUfHyByrL7jyh6Y8jq_KpwGFcwEtIsesL6iODD-gYHIgYfzSMzOb_flXyE5IY9ReSTZOOEAHc_RMhEYjTfDDRdDamoH4vlS-96-d-FVdVwx8CJSUJbdI8p6YsoY3DoVcHk9VCJcjVr2yejU74Zcs0ub4sPAGtZCjIe5besdKx40QOjyGihzx2e-FCVKLi0VMv1pu2VA8kyEr0JzZBaR5oSXOaMlbj6CQ2o-e725kNfbLUzZld94g6zNvW1NiLVu7XC-yT4=s470-no)
This above image might be difficult to classify because the traffic sign represented above has a **square boundary** which makes it similar to other traffic signs of the same shape.
![No Vehicles](https://lh3.googleusercontent.com/2xmhCyJQbmQm7i-PfXM-l2d8MZx-kCI_wyp5ru7l6LX7fcKpTd8A8MNeqFvjRLfdPA-APEhLOZReexazYPKek4FuxodQ56urTjhSNm8NaNguUejTOtYNsfazq1n3lFnMvLAVz8LkQVtL8dXgDjpVzIw_brvmrjUGVLIVSOLwgthQU14JiBEz25SkW8KhR64JwJIp7uikEniU5ClUSNCvq3lQCOs5bjCbBGtmLSwz9XzWGL-fIIs7-P2EwEWluIANG3BxYvYqddwy4qDf_zQyhP9w42ejyFmrWOSNSsuR-5cDHcxt5tGN7qAd90vg75rYbe1pdWudYm9-YqvFWFqbil7kW6ZaHN_uV9ctBKUMRSmJ79R21fPHK9k0XpCAVB4A4jrHVY-1CrHUeDvPDmzL8_JEdB_4Oxj0Epz2VlMTJ-8dhFtKpfIJi5BlIHY2ivusAVF7nvtnxpluP_mU_zmqwZqtiI5Re9BxU6925IccIF5QlH_M7L4QIsDe1OAA9wlMS1m6opIIbO8EjiUnY37kt7_DLoMdWe9UAgYTMgEEwLZmD7ToFw4GoUdyX8_luPzSi2Fw5ugxYwY_1Hi0upLu3ofxcL9-eDJEla_Kt3LWOriuwXsQQ5R1Rg-1px1ZzN0qPnzxZwxcxAAa2KnB4TRb_eIQWBUi7_8=s99-no)
This above image might be difficult to classify because the circle is a bit skewed in the image.
![General Caution](https://lh3.googleusercontent.com/gQi-T7CKTEmrGRboH1dfmpXE9Z8GZZVUKDGTDc3Lfvk80ClK33QZlRR1oWht9KQB5PfH-teaAYzl0oeY3816s8JdYe4Zj9jQZDmLNA7EqqG3gpNdtmaWIKxYGYPH0NOUjnh4K_WOrgVPe3TkRU8LJ-IYXjelfrGssF8rXWZBGNhSXzO8QAzsTUmW2S3sD-BDgxKOZzFcV45jgoGfujuuffQnERojmi4X_hINau0UbrUZDiX0tU8bOvYh-UHkhaF7wSmCFbzynZubJGV9N3tcZOD7RcM1BVhAQZ9gM0zGoKsUlpaV5psID2DAhk2tVrdBsmKBdUQY6SWSescyUy0Ti8mHq7Gtv0XCr-5PzXTlCUT1njZzQw8qF2jiVn86PmD2z6qwyvVOqwSh6p8Jan0jSJrO3uZ8lohNivI68UcUYPb37ME-oNsgDfO-33dSgm5HvDNPjd6Ye3XG8jtxxjM-rwdMX_Db6AK0v1u57uIrNrs18H8xvVskVZ4-oWd5N1IHEVa74A7mpdNVCVCXDipwIgl0oMRiS1zBjgHhOZafiMCWbIXczMbuC2K6GbFo8JcqSlZGQV6PvJ2aiYxiGX5U4fIAVw6JYFI5NY9IwavQOUMx_Whw53A8Ksr6kmv2uV4_DLArUT1oqzfdKuYjgtTB4FyWCdJHIHs=s133-no) 
This above image might be difficult to classify because the traffic sign represented above has a **triangular boundary** which makes it similar to other traffic signs of the same shape. Such as the **Right of way** sign above.
Furthermore, the sign is a bit bent to the sideways making its classification difficult. Also, it has a black patch in the traffic sign boundary, which might result in false classification.
![No Passing/Overtake](https://lh3.googleusercontent.com/4WCXPhb5lMyGqmEfwOw3cwGJdKj7E9qK3TlzjEb3JF1PNmsaeoAldhQr7qZWoWORL6LCykOwHys3j4KWYrP2WNwTOmu5G-TOBvcY3X3_J7UZhMGxXFi-ojpvJNK-JozDEbRlHUgWV5CcG8JXy4wFnjGaIYb9fZeWQPJY92WjwszHwCasNsf7jaOcc5TOvCgZomEwHMeubfJS7gNRNs59pwaHvBDGPE-DbT06zTJXb7Y7gy2ezH3bS-qxZfVRMZPR0WveYNBU0FPXU3NBczJQ9bIUYLScUnalcMMaBcH_ZZSO6scOZlMXal3ik3_jNZrsTsIhOIYJv8xQrY9ubH5eCnEinPFPFG4orpTkEb4TLfoCnLYFakgQDumdGnknhsLD4Xyy5OnafijBrayRGkhuSPz-pZLK88xJbTiNZr2IL1vPKqkO-mnkGojCJ--G1D2ti2NmmhI5NBeHX8O0eKYELClgzgQgbGIxkQBPZ9QT12WGBogEd6MA4MUZDrtAX1Ymqsgj2flwP4gOB2FDveW0dV7V5ZzpV2_IirdEJQ4oWMtCJFCUKP-VRTYns2MoJtyJxJ5uJjakVxcyiMxFsKhF9YE8fQUd3FOAigBtxKsGZZQWEFylMZR1-qzaQbbmgC19Wts-NOzDl5wbUqGhoApZa8IdnKSbC_A=s98-no)
This above image might be difficult to classify because the traffic sign represented above has a **circular boundary** which makes it similar to other traffic signs of the same shape. Such as the **No Vehicles** sign above.
Furthermore, the sign has smudges on it which might result in a non-accurate classification.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing      		| Stop   									    | 
| Right of way     		| Right of way 									|
| Priority road			| Priority road									|
| No vehicles	      	| No vehicles					 				|
| General Caution		| General Caution      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of **0.949**

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

![Softmax Probabilities](https://lh3.googleusercontent.com/zdvhMyz6ORIOxtiC6Q7HI1AMUJoWBQJ4YR8zXq3cvT0pNx2o9VlD6SeHxY8bbTV3achaf94uICFIQIteWSJ-qEKl8d-8FjDzOph2-aSlJrBeu68Ol3inK9zV2kIdBDv8Yr89qgIv9i9BmI5u5Ag02BFFLL3HhZ1C-q-CM68lsLZmZ_FNNQosT2ypiYEBgKs_MYz0i15qZjyyBiTDDQ7iLKdV3389pu8RWcLSigR0VtfZYHBEaPDy7rosOQm-CSMHIY2dKNLpewz5m8n9Vod0x8k0_BO03ph2SL9L8OyzyKsxo5kFyqeT0t_3hW2sYfrIb0zSiVBpUAkS9vD86lR59UikznCTbImpvXNGTXuUC1q2qXptCgY_RMcMXelxIKJ6fNGKYNzTOHqqa0uWlaGVLcowq8DypP_-B2Qz9D__ADj5pnStQZF5ahAylHSsTXd0R-tJR0DHwnH-4pYMfb-2i_NfltLkSkMKoUGpExkqzYQD-EZlacl3kmveBy2OiodXgamEjsJv9FBr_XDQtprFZcqIeQ-NBVvVaABssfH7KHTD8N2CRpZFhQ09tKKDqmpeiCgCR4PWrKp72L7J9gWzdo6CmGHA09QwPiVxig3fU14Wt20iRgeDeoJKKgXSqKl-jghFp1kPlIJTbhRf9vQp4zCgKI5ZYQM=w276-h316-no)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


