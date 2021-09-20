# Support Vector Machine

Support Vector Machine is a Supervised Machine Learning Algorithm. It is used for both the purposes of Regression and Classification. However, SVM is mostly used in Classification Problems.

Suppose, there are n features in a dataset and the data points are plotted in an n-dimensional space. The Support Vector Machine then finds the best hyperplane, which is a boudary, to perform Classification. The Support vectors are the co-ordinates of the data points, that lie closest to the hyperplane.

<p align="center">
  <img src = "https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_1.png">
  <img src = "https://miro.medium.com/max/1166/1*iUPqc_nTwe_O_GB4F6qGXw.png">
</p>
  
## Hyperplane

A hyperplane in an n-dimensional Euclidean space is a decision boundary, n-1 dimensional subset of that space that divides the n-dimesional space into two disconnected parts. In simple words, Hyperplanes can be considered decision boundaries that classify data points into their respective classes in a multi-dimensional space. Data points falling on either side of the hyperplane can be attributed to different classes. 

Three examples of hyperplane are :

1. A hyperplane in 1D is a point.

2. A hyperplane in 2D is a line. 

<p align="center">
  <img src = "https://miro.medium.com/max/1120/0*IpPnPw9NX3n-VgJY.png" width = 400 height = 400>
</p>

3. A hyperplane in 3D is a plane.

<p align="center">
  <img src = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQwAGq9fZLrN27AyLVQyvz0Ki4TgaIqC-1o6o025HrwhxNcL_V9WHOy07AHbtJbJ4SIGpg&usqp=CAU" width = 400 height = 400>
</p>
  
A hyperplane may be linear or non-linear. 

1. For linear data

<p align="center">
  <img src = https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVogeHiAvdIBor0sVprImY5sISjYkZHIGccA&usqp=CAU width = 400 height = 400>
</p>

2. For non-linear data

<p align="center">
  <img src = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSLjMazcTjjnu9m44QAj8I-TlXQkEXgjC_23A&usqp=CAU" width = 400 height = 400>
</p>

However, if we transform the values in the above plot with the equation <img src="https://render.githubusercontent.com/render/math?math=\Large z^2 = x^2 %2B y^2">, then the data points will become linearly seperable.

<p align="center">
  <img src="https://github.com/Adi-ds/Private-Repo/blob/main/images/hyperplane.png" width = 800 height = 500> 
</p>

### Equation of Hyperplane

In D dimensional space, the hyperplane would always be D -1 operator.

#### Case I

Let us consider the hyperplane in a 2D space.

<p align="center">
  <img src = https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVogeHiAvdIBor0sVprImY5sISjYkZHIGccA&usqp=CAU width = 400 height = 400>
</p>

The equation of the straight line dividing the two partitions is <img src="https://render.githubusercontent.com/render/math?math=\Large y = mx %2B c">
<img src="https://render.githubusercontent.com/render/math?math=\Large \implies mx %2B c - y = 0"> <img src="https://render.githubusercontent.com/render/math?math=\Large \implies w^T X %2B c = 0">

So, the hyperplane equation becomes <img src="https://render.githubusercontent.com/render/math?math=\Large w^TX %2B c = 0">, where X = (x,y) and w = (a,-1).

#### Case II

Let us consider the hyperplane in a 3D space.

<p align="center">
  <img src = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQwAGq9fZLrN27AyLVQyvz0Ki4TgaIqC-1o6o025HrwhxNcL_V9WHOy07AHbtJbJ4SIGpg&usqp=CAU" width = 400 height = 400>
</p>

So, the equation of the plane seperating the classes is <img src="https://render.githubusercontent.com/render/math?math=\Large w_1x_1 %2B w_2x_2 %2B w_3x_3 %2B d = 0 \implies  w^Tx %2B d= 0">

Hence the hyperplane is <img src="https://render.githubusercontent.com/render/math?math=\Large w^Tx %2B d= 0">, where <img src="https://render.githubusercontent.com/render/math?math=\Large w = (w_1,w_2,w_3)"> and <img src="https://render.githubusercontent.com/render/math?math=\Large x = (x_1,x_2,x_3)">

#### Case III

Now, let us consider an n-dimensional plane. The Equation of the hyperplane will be given by -

<img src="https://render.githubusercontent.com/render/math?math=\Large b %2B b_1 x_1 %2B b_2 x_2 %2B .... %2B b_n x_n = 0">

<img src="https://render.githubusercontent.com/render/math?math=\Large \implies w^Tx %2B b = 0">, where <img src="https://render.githubusercontent.com/render/math?math=\Large w = (b_1,b_2,....,b_n)"> and <img src="https://render.githubusercontent.com/render/math?math=\Large x = (x_1,x_2,....,x_n)">

Here, b is the intercept and bias term of the hyperplane equation.

### Optimal Hyperplane

An infinite number of hyperplanes can be obtained in a classification problem, as shown in the following diagram.

<p align = "center">
  <img src = "https://miro.medium.com/max/479/1*62qYJYpLZu4KQuyQIpy5pg.png" width = 300 height = 300>
 </p>

SVM tries to find the Optimal hyperplane, which divides the data points very well. In other words, the optimal hyperplane is that hyperplane, which is right at the center where the distance is maximum from the closest points and give the least test errors further.

<p align = "center">
  <img src = "https://miro.medium.com/max/552/1*q7Tr-GNbm5HM7T3d1HR-Uw.png">
 </p>
 
Let’s assume that solid deep-green line in the above figure is optimal hyperplane and two dotted line are some hyperplanes, which are passing through the support vectors. Then distance between the hyperplanes and optimal hyperplane is know as margin. Margin is an area which do not contains any data points. So, while choosing optimal hyperplane, the hyperplane which is highest distance from the support vectors is to be choosen. If optimal hyperplane is very close to data points then margin will be very small and it will generalize well for training data but when an unseen data will come it will fail to generalize well. So our goal is to maximize the margin so that our classifier is able to generalize well for unseen instances, i.e, to choose an optimal hyperplane which maximizes the margin.
