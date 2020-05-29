# CS294-158-SP19

My Pytorch implementation of Berkeley Deep Unsupervised Learning from Spring 2019 <br>
Only notable results have been included


## Homework 1 Results

### Exercise 1.2
### MADE network on 2d data:
![img](Hw1/Figures/Figure_6.png)

### PixelCNN results
![img](Hw1/Figures/Figure_9_Adamax100.png)

## Homework 2 Results
### Original Data, note colors
![img](Hw2/Figures/Figure_1.png)

### Autoregressive Flow results
![img](Hw2/Figures/Figure_2.png)

### Autoregressive Flow Latent Space
![img](Hw2/Figures/Figure_3.png)

### Simple Real NVP Results
![img](Hw2/Figures/Figure_6.png)

### Simple Real NVP Latent and Generated Samples
![img](Hw2/Figures/Figure_5.png)

### Simple Real NVP Latent Colored
![img](Hw2/Figures/Figure_7.png)

## Homework 3 Results
### Model Training Curves. <br>
Note that scalar variance data set 1 KL is almost 0
![img](Hw3/Figures/Figure_2-1.png)

### Results on Data set 1
Note how mean of vector variance model is centered at 0
![img](Hw3/Figures/Figure_4ds1-1.png)

### Latent Space of Data set 1 Models
Top: Data colored.
Note, how the latent of vector variance is completely scrambled, no data is stored. Reconstructions are similar. This shows latent is not used. This is due to the data being able to be completely modeled by the decoder with no help of the latent.
![img](Hw3/Figures/latent_visualization_ds1-1.png)

### Results on Data set 1
![img](Hw3/Figures/Figure_4ds2-1.png)

### Latent Space of Data set 1 Models
This time both models behave the same due to the multivariate diagonal covariance gaussian being insufficient to model the data alone.
![img](Hw3/Figures/latent_visualization_ds2-1.png)

### IWAE Training
![img](Hw3/Figures/Figure_5-1.png)

### IWAE Results
![img](Hw3/Figures/Figure_6-1.png)

### IWAE Latent space
![img](Hw3/Figures/Figure_7-1.png)

### SVHN Training Curve
![img](Hw3/Figures/Figure_8-1.png)

### SVHN Training Curve
![img](Hw3/Figures/Figure_8-1.png)

### SVHN Results
![img](Hw3/Figures/Figure_9_FullLoss-1.png)

### SVHN Interpolations
![img](Hw3/Figures/Figure_10-1.png)

### WGAN Training Curves
![img](Hw4/figures/Training_curves-1.png)

### WGAN Results
![img](Hw4/figures/Best_samples-1.png)







