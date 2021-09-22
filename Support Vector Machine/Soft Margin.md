The derivation of the SVM as presented in [Hard Margin](https://github.com/Adi-ds/Titanic_Kaggle/blob/main/Support%20Vector%20Machine/Hard%20Margin.md) section assumed that the data is linearly separable. Often data is not linearly separable and there exists no hyperplane so that it can classify among the variable points separately. 

<p align = "center">
  <img src = "https://miro.medium.com/max/700/1*tl3dQaEConfFoTVM0amXHA.png">
</p>

In fact, even if the data is linearly separable, it might still not be beneficial to use the hard-margin SVM because outliers can have a large impact on the location of the hyperplane. If it is strictly imposed that hyperplane stays as far as from the closest data points,then the data points will be linearly separable but there will be also narrow margin, in that case model will be extremely sensitive to noisy data points , and model will lack generalization capability i.e. chance of under-fitting will increases. Also, it will only work with linearly separable data-points.

<p align = "center">
  <img src = "https://github.com/Adi-ds/Private-Repo/blob/main/images/svm.png">
</p>

To avoid these issues it is preferable to use more flexible model with some margin violation. It is better to have large margin, even though some constraints is violated. Margin violation means choosing an hyperplane, which can allow some data points to stay in either in between the margin area or in the incorrect side of hyperplane, which is contrast to hard margin classification task. In other words, instead of trying to maximize the distance between the hyperplane and the closest data points while keeping all points correctly classified, we try to maximize the margin but allow data points to be wrongly classified but discouraging this by adding penalization to the objective function.This type of classification are called as soft margin classification.

Now, let us consider a dataset <img src="https://render.githubusercontent.com/render/math?math=\Large S = \{(\vec{x_i},y_i), i = 1,2,....,n, y_i \in \{-1,+1\}\}">.

To relax the constrains of the equation slightly to allow the margin violation to occur slack variables, <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_1, \xi_2, ....., \xi_n"> are introduced, so that the new hyperplane equations are :

<p align = "center">
  <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{w}^T\vec{x_i} %2B b \geq 1 - \xi_i"> for <img src="https://render.githubusercontent.com/render/math?math=\Large y_i = %2B 1">
</p>


<p align = "center">
  <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{w}^T\vec{x_i} %2B b \leq -1 %2B \xi_i"> for <img src="https://render.githubusercontent.com/render/math?math=\Large y_i = - 1">
</p>

Combining the above two equations, it can be written as <img src="https://render.githubusercontent.com/render/math?math=\Large y_i\left(\vec{w}^T\vec{x_i} %2B b\right) - 1 %2B \xi_i \geq 0, \forall i = 1,2,.....,n">.

Here,
- <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i = 0"> in case of no error, i.e., for the points are in their ideal locations.
- <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i > 0"> for the violations from ideal locations.
  - <img src="https://render.githubusercontent.com/render/math?math=\Large 0 < \xi_i < 1">, for the points which lie on the correct side of the hyperplane, but within the margin.
  - <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i > 1">, for the points which lie on the wrong side of the hyperplane.

Hence, <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i \geq 0 , \forall i = 1,2,....,n">.

The values of <img src="https://render.githubusercontent.com/render/math?math=\Large {\xi_i}'s"> are shown in the following figure.
<p align = "center">
  <img src="https://github.com/Adi-ds/Private-Repo/blob/main/images/svm2.png">
</p>

Now, in case of soft-margin classifier, <img src="https://render.githubusercontent.com/render/math?math=\Large \sum_{i=1}^n \xi_i"> is also to be minimized, to reduce the total proportional amount by which predictions fall on the wrong side of their margin.

Hence, to maximize the margins and minimize the mistakes, the primary optimization problem is written as :

<p align = "center">
  Minimize <img src="https://render.githubusercontent.com/render/math?math=\Large { \frac {||\vec{w}||^2}{2} } %2B C{\sum_{i=1}^n \xi_i}"> subject to <img src="https://render.githubusercontent.com/render/math?math=\Large y_i\left(\vec{w}^T\vec{x_i} %2B b\right) - 1 %2B \xi_i \geq 0">, <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i \geq 0 , \forall i = 1,2,.....,n">.
</p>

Here, <img src="https://render.githubusercontent.com/render/math?math=\Large C \geq 0"> is the penalty term. From this,  it can be identified that if <img src="https://render.githubusercontent.com/render/math?math=\Large C = 0">, the hard-margin problem is recovered.

&nbsp;&nbsp;&nbsp;&nbsp; <em><b>Significance of <img src="https://render.githubusercontent.com/render/math?math=\Large C"></b>
>Let us consider an optimization problem of the form <img src="https://render.githubusercontent.com/render/math?math=\Large \min _{\mathbf{x}, \mathbf{y}} \alpha f(\mathbf{x}) %2B \beta g(\mathbf{y}) \quad \text { s.t. } \quad(\mathbf{x}, \mathbf{y}) \in D"> where <img src="https://render.githubusercontent.com/render/math?math=\Large \alpha, \beta>0"> are some constants. To make the objective as small as possible, <img src="https://render.githubusercontent.com/render/math?math=\Large f"> and  <img src="https://render.githubusercontent.com/render/math?math=\Large g"> are to be balnced somehow. Choosing <img src="https://render.githubusercontent.com/render/math?math=\Large x"> such that <img src="https://render.githubusercontent.com/render/math?math=\Large f"> is small might constrain us to choose <img src="https://render.githubusercontent.com/render/math?math=\Large y"> such that <img src="https://render.githubusercontent.com/render/math?math=\Large g"> becomes larger, and vice versa. If <img src="https://render.githubusercontent.com/render/math?math=\Large \alpha"> is much larger then <img src="https://render.githubusercontent.com/render/math?math=\Large \beta">, then it is 'more beneficial' to make <img src="https://render.githubusercontent.com/render/math?math=\Large f"> small, at the expense of making <img src="https://render.githubusercontent.com/render/math?math=\Large g"> a bit larger. The same holds the other way around.

> In case of SVM, there are two two functions, <img src="https://render.githubusercontent.com/render/math?math=\Large \|\mathbf{\vec{w}}\|^{2}"> and  <img src="https://render.githubusercontent.com/render/math?math=\Large \sum_{i=1}^n \xi_i"> and  <img src="https://render.githubusercontent.com/render/math?math=\Large \alpha = 1"> and  <img src="https://render.githubusercontent.com/render/math?math=\Large \beta = C">. If  <img src="https://render.githubusercontent.com/render/math?math=\Large C"> is much smaller than 1, then it is beneficial to make <img src="https://render.githubusercontent.com/render/math?math=\Large \|\mathbf{\vec{w}}\|^{2}"> small. If <img src="https://render.githubusercontent.com/render/math?math=\Large C"> is much larger than 1, then it is the other way around. It turns out that <img src="https://render.githubusercontent.com/render/math?math=\Large \sum_{i=1}^n \xi_i">, since <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i \geq 0">, happens to be exactly <img src="https://render.githubusercontent.com/render/math?math=\Large ||\xi||_{1}"> meaning that the entries <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_{i}"> become small. Moreover, it is well-known that attempting to minimize the <img src="https://render.githubusercontent.com/render/math?math=\Large l_1"> norm promotes sparsity, meaning that as C increases, more and more entries of ξ become zero.

>> So to sum up, <img src="https://render.githubusercontent.com/render/math?math=\Large \mathbf \xi_i's"> are the deviations that is allowed from the inequality <img src="https://render.githubusercontent.com/render/math?math=\Large y_{i}(\vec{w}^T\vec{x_i} %2B b) \geq 1">. When <img src="https://render.githubusercontent.com/render/math?math=\Large C"> is large, minimizing <img src="https://render.githubusercontent.com/render/math?math=\Large { \frac {||\vec{w}||^2}{2} } %2B C{\sum_{i=1}^n \xi_i}"> means that <img src="https://render.githubusercontent.com/render/math?math=\Large \mathbf \xi_i's"> will be small, since their sum has a large weight. When <img src="https://render.githubusercontent.com/render/math?math=\Large C"> is small, it means that their sum has a small weight, and at the minimum <img src="https://render.githubusercontent.com/render/math?math=\Large \mathbf \xi_i's"> may be larger, allowing more deviation from the above inequality. When <img src="https://render.githubusercontent.com/render/math?math=\Large C"> is extremely large, the only way to minimize the objective is to make the deviations extremely small, bringing the result close to hard margin SVM.
</em>
  
The Primal problem of the soft-margin SVM leads to the following Lagrangian:

<p align = "center">
  <img src="https://render.githubusercontent.com/render/math?math=\Large L\left(\vec{w},b,\vec{\xi}\right)=\frac{\left|\left|\vec{w}\right|\right|^2}{2} %2B C\sum_{i=1}^{n}\xi_i - \sum_{i=1}^{n}\lambda_i(y_i(\vec{w}^T\vec{x_i} %2B b)-1 %2B \xi_i) - \sum_{i=1}^{n}\mu_i\xi_i..............................eq(1)">
</p>

Now the first order derviatives are :

<img src="https://render.githubusercontent.com/render/math?math=\Large \frac{\delta L}{\delta\vec{w}}=\vec{w}-\sum_{i=1}^n\lambda_iy_ix_i = 0 \implies \vec{w} = \sum_{i=1}^n\lambda_iy_ix_i..............................eq(2)">

<img src="https://render.githubusercontent.com/render/math?math=\Large \frac{\delta L}{\delta b}=-\sum_{i=1}^n\lambda_iy_i = 0 \implies \sum_{i=1}^n\lambda_iy_i = 0............................................eq(3)">
     
<img src="https://render.githubusercontent.com/render/math?math=\Large \frac{\delta L}{\delta\xi_i}=\ C-\mu_i-\lambda_i = 0 \implies C = \mu_i %2B \lambda_i.....................................................eq(4)">

So from eq(1) we get -

<img src="https://render.githubusercontent.com/render/math?math=\Large L\left(\vec{w},b,\vec{\xi}\right) = \begin{array}{l}\frac{1}{2}\vec{w}^T\vec{w}\ %2B \ \sum_{i=1}^n\left(C-\mu_i-\lambda_i\right)\end{array}\xi_i-\sum_{i=1}^n\lambda_iy_i\vec{w}^T\vec{x_i}- b\sum_{i=1}^n\lambda_iy_i %2B \sum_{i=1}^n\lambda_i ">

<img src="https://render.githubusercontent.com/render/math?math=\Large \implies L\left(\vec{w},b,\vec{\xi}\right) = {\frac {{\sum_{i = 1}^n}(\lambda_j y_j \vec{x_j})^T(\lambda_j y_j \vec{x_j})}{2}} - {\sum_{j = 1}^n}{\lambda_j y_j ( ({\sum_{i = 1}^n}\lambda_i y_i \vec{x_i})^T \vec{x_j} )} %2B {\sum_{i = 1}^n} \lambda_i ">[ using eq(2), eq(3), eq(4)]

<img src="https://render.githubusercontent.com/render/math?math=\Large \implies L\left(\vec{w},b,\vec{\xi}\right) = {\frac { {\sum_{i = 1}^n}{\sum_{j = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j}}{2}} - {\sum_{i = 1}^n}{\sum_{j = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j} %2B {\sum_{i = 1}^n}{\lambda_i}\:\:\:[\because {\sum_{i = 1}^n} {\lambda_i y_i} = 0]">


<img src="https://render.githubusercontent.com/render/math?math=\Large \implies L\left(\vec{w},b,\vec{\xi}\right) = {\sum_{i = 1}^n}{\lambda_i} - {\frac { {\sum_{i = 1}^n}{\sum_{j = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j}}{2}}">

And the KKT conditions are :

- <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{w} = \sum_{i=1}^n\lambda_iy_ix_i">...................................eq(5)
- <img src="https://render.githubusercontent.com/render/math?math=\Large \sum_{i=1}^n\lambda_iy_i = 0">.........................................eq(6)
- <img src="https://render.githubusercontent.com/render/math?math=\Large C = \mu_i %2B \lambda_i">................................................eq(7)
- <img src="https://render.githubusercontent.com/render/math?math=\Large \lambda_i(y_i(\vec{w}^T\vec{x_i} %2B b)-1 %2B \xi_i) = 0">...........eq(8)
- <img src="https://render.githubusercontent.com/render/math?math=\Large y_i(\vec{w}^T\vec{x_i} %2B b)-1 %2B \xi_i \geq 0">...................eq(9)
- <img src="https://render.githubusercontent.com/render/math?math=\Large \mu_i \xi_i = 0">......................................................eq(10)
- <img src="https://render.githubusercontent.com/render/math?math=\Large \mu_i \geq 0">...........................................................eq(11)
- <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i \geq 0">...........................................................eq(12)

Hence we get the following dual problem :

<p align = "left">
  <img src="https://render.githubusercontent.com/render/math?math=\Large {max_{\vec{\lambda}}}\:\:W(\vec{\lambda}) = {\sum_{i = 1}^n}{\lambda_i} - {\frac {1}{2}}{\sum_{i = 1}^n}{\sum_{j = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j}">
    <text>  such that </text>
      <img src="https://render.githubusercontent.com/render/math?math=\Large {\sum_{i = 1}^n} {\lambda_i y_i} = 0 ">
        <text> , </text>
          <img src="https://render.githubusercontent.com/render/math?math=\Large 0 \leq \lambda_i \leq C\:\forall\:i = i,2,.....,n">
</p>

Now, let us look at the terms individually.

![first term](https://github.com/Adi-ds/Private-Repo/blob/main/images/equatio.png)

<img src="https://equatio-api.texthelp.com/png/%5Cbegin%7Barray%7D%7Bl%7D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7B1%3D1%7D%5En%5Clambda_i%5Clambda_jy_iy_jx_i%5ETx_j%5C%5C%0D%0A%3D%5Cfrac%7B1%7D%7B2%7D%5Cleft(%5Clambda_%7B1%5C%20%7D%5Clambda_2%5C%20...%5C%20%5Clambda_n%5Cright)%5Cbegin%7Bpmatrix%7Dy_1y_1x_1%5ETx_1%26y_1y_2x_1%5ETx_2%26..........%26y_1y_nx_1%5ETx_n%26%5C%5C%0D%0Ay_2y_1x_2%5ETx_1%26y_2y_2x_2%5ETx_2%26..........%26y_2y_nx_2%5ETx_n%26%5C%5C%0D%0A..............%26..............%26..........%26..............%26%5C%5C%0D%0A..............%26..............%26..........%26..............%26%5C%5C%0D%0A..............%26..............%26..........%26..............%26%5C%5C%0D%0Ay_ny_1x_n%5ETx_1%26y_ny_2x_n%5ETx_2%26..........%26y_ny_nx_n%5ETx_n%26%5Cend%7Bpmatrix%7D%5Cbegin%7Bpmatrix%7D%5Clambda_1%5C%5C%0D%0A%5Clambda_2%5C%5C%0D%0A.%5C%5C%0D%0A.%5C%5C%0D%0A.%5C%5C%0D%0A%5Clambda_n%5C%5C%0D%0A%5Cend%7Bpmatrix%7D%5C%5C%0D%0A%3D%5Cfrac%7B1%7D%7B2%7D%5Cvec%7B%5Clambda%7D%5ETQ%5Cvec%7B%5Clambda%7D%5Cend%7Barray%7D?height=286">

<img src="https://render.githubusercontent.com/render/math?math=\Large \sum_{i=1}^n\lambda_iy_i=\lambda_1+\lambda_2+....+\lambda_n=\ \left(\lambda_1\ \lambda_{2\ }....\ \lambda_n\right)\left[\begin{matrix}y_1\\y_2\\.\\.\\.\\y_n\end{matrix}\right] = \vec{\lambda}^T\vec{y}">

Now, we can write the dual problem as 

<img src = "https://equatio-api.texthelp.com/png/K%5C%20%3D%5C%20%5Cvec%7B%5Clambda%7D%5ET-%5C%20%5Cfrac%7B1%7D%7B2%7D%5Cvec%7B%5Clambda%7D%5ETQ%5Clambda%2B%5Cbeta%5Cvec%7B%5Clambda%7D%5ET%5Cvec%7By%7D?height=49"> where <img src="https://render.githubusercontent.com/render/math?math=\Large \beta"> is the Lagrange Multiplier.

So, <img src = "https://equatio-api.texthelp.com/png/%5Cbegin%7Barray%7D%7Bl%7D%5Cnabla_%7B%5Cvec%7B%5Clambda%7D%7DK%5C%20%3D%5C%20%5Cvec%7B1%7D-%5C%20Q%5Cvec%7B%5Clambda%7D%5C%20%2B%5C%20%5Cbeta%5Cvec%7By%7D%5C%20%3D%5C%200%5C%20%5CRightarrow%5C%20Q%5Cvec%7B%5Clambda%5C%20%7D%3D(%5Cvec%7B1%7D%5C%20%2B%5C%20%5Cbeta%5Cvec%7By%7D)%5C%20%5CRightarrow%5C%20%5Cvec%7B%5Clambda%7D%5E%7B*%7D%5C%20%3D%20%5C%20Q%5E%7B-1%7D(%5Cvec%7B1%7D%5C%20%2B%5C%20%5Cbeta%5Cvec%7By%7D)%5Cend%7Barray%7D?height=37">.

Here, <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{\lambda}^{*}"> is the obtained value of <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{\lambda}">.

Therefore, the obtained value of <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{w}"> will be <img src = "https://equatio-api.texthelp.com/png/w%5E*%3D%5C%20%5Csum_%7Bi%3D1%7D%5En%5Clambda_i%5E*y_ix_i?height=73">. 

Clearly, the optimal <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{w^{*}}"> is a function of <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{\lambda}^{*}">.

Therefore, the optimal solution of <img src="https://render.githubusercontent.com/render/math?math=\Large b"> is given by <img src = "https://render.githubusercontent.com/render/math?math=\Large b^*\ =\ -\ \frac {\max_{i:y_i=-1}\vec{w^*}^T\vec{x_i}\ %2B \min_{i:y_i=1}\vec{w^*}^T\vec{x_i} }{2}">.

Hence, the hyperplane equation becomes, 

<p align = "center">
  <img src="https://render.githubusercontent.com/render/math?math=\Large {w^*}^Tx %2B b^* = ( {\sum_{i = 1}^n}{\lambda_i^*}y_ix_i )^Tx %2B b^* = {\sum_{i = 1}^n}{\lambda_i^*}y_i(x_i^Tx) %2B b^*">
</p>

Moreover, the <img src="https://render.githubusercontent.com/render/math?math=\Large {\lambda_i}^*"> ’s will all be zero except for the support vectors. Thus, many of the terms in the sum above will be zero, only the products between x and the support vectors (of which there is often only a small number) in order to make prediction.

Now, considering the KKT conditions
- If <img src="https://render.githubusercontent.com/render/math?math=\Large y_i\left(\vec{w}^T\vec{x} %2B b\right)-1 %2B \ \xi_i > 0">, then <img src="https://render.githubusercontent.com/render/math?math=\Large \lambda_i = 0"> and <img src="https://render.githubusercontent.com/render/math?math=\Large \lambda_i = C - \mu_i = 0 \implies \mu_i = C \neq 0">, then <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i = 0"> and <img src="https://render.githubusercontent.com/render/math?math=\Large y_i(\vec{w}^T\vec{x} %2B) > 1">.
- Otherwise, <img src="https://render.githubusercontent.com/render/math?math=\Large y_i\left(\vec{w}^T\vec{x} %2B b\right)-1 %2B \ \xi_i > 0">, then <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i = 1 - y_i\left(\vec{w}^T\vec{x} %2B b\right)">

Consequently, if <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i^*"> be the optimized value of <img src="https://render.githubusercontent.com/render/math?math=\Large \xi_i">, then we will get - 

<p align = "center">
  <img src="https://render.githubusercontent.com/render/math?math=\Large  \xi_i^{*} =  \begin{pmatrix} 0 & if\ (1 - y_i(\vec{w^*}^T\vec{x_i} = b^*)) < 0 \\ (1 - y_i(\vec{w^*}^T\vec{x_i} = b^*)) &  if\ (1 - y_i(\vec{w^*}^T\vec{x} + b^*)) \geq 0 \end{pmatrix}">
</p>

