Consider a training set <img src="https://render.githubusercontent.com/render/math?math=\Large S = \{(\vec{x_i},y_i), i = 1,2,....,n, y_i \in \{-1,+1\}\}">.

From the previous section it is clear that, we have to find a decision boudary, i.e, an optimal hyperplane, which maximizes the margin. In other words, the goal is to maximize the minimum distance. Specifically, this will result in a classifier that separates the positive and the negative training examples with a “gap” (geometric margin), as shown below. 

<p align = "center">
  <img src = "https://github.com/Adi-ds/Private-Repo/blob/main/images/image.png" width = 400 height = 300>
</p>

Now, consider the following figure.

<p align = "center">
  <img src = "https://miro.medium.com/max/511/1*uqqLm5bSTY-pD4CmW3SWew.png">
</p>

In the figure, we have two hyperplanes <img src="https://render.githubusercontent.com/render/math?math=\Large H_1"> and <img src="https://render.githubusercontent.com/render/math?math=\Large H_2"> passing through the support vectors of -1 and +1 class respectively.

<p align = "center">
  <img src="https://render.githubusercontent.com/render/math?math=\Large H_1 : \vec{w}^T\vec{x} %2B b = -1">
</p>

<p align = "center">
  <img src="https://render.githubusercontent.com/render/math?math=\Large H_2 : \vec{w}^T\vec{x} %2B b = 1">
</p>  

The distance between <img src="https://render.githubusercontent.com/render/math?math=\Large H_1"> and the origin is <img src = "https://render.githubusercontent.com/render/math?math=\Large \frac {b %2B 1}{|\vec{w}|}"> and the distance between <img src="https://render.githubusercontent.com/render/math?math=\Large H_2"> and origin is <img src="https://render.githubusercontent.com/render/math?math=\Large \frac {b-1}{|\vec{w}|}">. So, the margin bewteen the hyperplanes is, <img src="https://render.githubusercontent.com/render/math?math=\Large M = {\frac {b %2B 1}{|\vec{w}|}} - {\frac {b-1}{|\vec{w}|}} = {\frac {2}{|\vec{w}|}}">. However, here <img src="https://render.githubusercontent.com/render/math?math=\Large M"> is twice the margin. As, optimal hyperplane maximize the margin, then the SVM objective is boiled down to fact of maximizing the term <img src="https://render.githubusercontent.com/render/math?math=\Large \frac {1}{|\vec{w}|}">.

Now, <img src="https://render.githubusercontent.com/render/math?math=\Large max \frac {1}{|\vec{w}|}"> can also be written as <img src="https://render.githubusercontent.com/render/math?math=\Large  min\:|\vec{w}|">. 

Again, <img src="https://render.githubusercontent.com/render/math?math=\Large L_2"> optimization is more stable than <img src="https://render.githubusercontent.com/render/math?math=\Large L_1"> optimization. 

So we get the Primal form of optimization problem, which can be written as

<p align = "center">
  <img src="https://render.githubusercontent.com/render/math?math=\Large  min\:{\frac {||\vec{w}||^2}{2}}">
    <text>such that</text>
      <img src="https://render.githubusercontent.com/render/math?math=\Large y_i(\vec{w}^T\vec{x_i} %2B b)-1 \geq 0">
        <text>for</text>
          <img src="https://render.githubusercontent.com/render/math?math=\Large i = 1,2,.....,n"> 
</p>

Clearly, SVM optimization problem is a case of constrained optimization problem. So, here Lagrange Multiplier Method will be used. Lagrange method is required to convert constrained optimization problem into unconstrained optimization problem. The goal of above equation to get the optimal value for <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{w}"> and <img src="https://render.githubusercontent.com/render/math?math=\Large b">.

So, using Lagrange multipliers <img src="https://render.githubusercontent.com/render/math?math=\Large {\lambda}_i, i = 1,2,.....,n"> the problem can be written as

<img src="https://render.githubusercontent.com/render/math?math=\Large L = {\frac {||\vec{w}||^2}{2}} - {\sum_{i = 1}^n} {\lambda_i (y_i (\vec{w}^T\vec{x_i} %2B b)-1)}..................................................................eq.1">

<img src="https://render.githubusercontent.com/render/math?math=\Large \implies L = {\frac {||w||^2}{2}} - {\sum_{i = 1}^n} {\lambda_i y_i (\vec{w}^T\vec{x_i} %2B b)} %2B {\sum_{i = 1}^n} \lambda_i .................................................eq.2">

So, we get <img src="https://render.githubusercontent.com/render/math?math=\Large \bigg\{ \begin{matrix} {\frac {\delta L}{\delta \vec{w}}} = \vec{w} - {\sum_{i = 1}^n} {\lambda_i y_i \vec{x_i}} = 0\\ {\frac {\delta L}{\delta b}} = {\sum_{i = 1}^n} {\lambda_i y_i} = 0 \end{matrix}"><img src="https://render.githubusercontent.com/render/math?math=\Large \implies \bigg\{ \begin{matrix} \vec{w} = {\sum_{i = 1}^n} {\lambda_i y_i \vec{x_i}} \\ {\sum_{i = 1}^n} {\lambda_i y_i} = 0 \end{matrix}">

Putting <img src="https://render.githubusercontent.com/render/math?math=\Large w = {\sum_{i = 1}^n} {\lambda_i y_i \vec{x_i}}"> in eq.2 we get -

<img src="https://render.githubusercontent.com/render/math?math=\Large L = {\frac {{\sum_{i = 1}^n}||{\lambda_i y_i \vec{x_i}}||^2}{2}} - {\sum_{j = 1}^n}{\sum_{i = 1}^n}{\lambda_j y_j ( (\lambda_i y_i \vec{x_i})^T \vec{x_j} %2B b)} %2B {\sum_{i = 1}^n} \lambda_i ">

<img src="https://render.githubusercontent.com/render/math?math=\Large \implies L = {\frac {{\sum_{i = 1}^n}(\lambda_j y_j \vec{x_j})^T(\lambda_j y_j \vec{x_j})}{2}} - {\sum_{j = 1}^n}{\sum_{i = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j} %2B b{\sum_{j = 1}^n}{\sum_{i = 1}^n}{\lambda_j y_j} %2B {\sum_{j = 1}^n}{\lambda_j}">

<img src="https://render.githubusercontent.com/render/math?math=\Large \implies L = {\frac { {\sum_{i = 1}^n}{\sum_{j = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j}}{2}} - {\sum_{i = 1}^n}{\sum_{j = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j} %2B {\sum_{i = 1}^n}{\lambda_i}\:\:\:[\because {\sum_{i = 1}^n} {\lambda_i y_i} = 0]">


<img src="https://render.githubusercontent.com/render/math?math=\Large \implies L = {\sum_{i = 1}^n}{\lambda_i} - {\frac { {\sum_{i = 1}^n}{\sum_{j = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j}}{2}}">

Now, the constraint for this problem was <img src="https://render.githubusercontent.com/render/math?math=\Large y_i(\vec{w}^T\vec{x_i} %2B b)-1 \geq 0">. So, <img src="https://render.githubusercontent.com/render/math?math=\Large \lambda_i \geq 0\:\forall\:i = i,2,.....,n"> otherwise the constraint will not remain considered in the problem.

Hence we obtain the following dual problem :

<p align = "left">
  <img src="https://render.githubusercontent.com/render/math?math=\Large {max_{\vec{\lambda}}}\:\:W(\vec{\lambda}) = {\sum_{i = 1}^n}{\lambda_i} - {\frac {1}{2}}{\sum_{i = 1}^n}{\sum_{j = 1}^n}{\lambda_i \lambda_j {\vec{x_i}}^T \vec{x_j} y_i y_j}">
    <text>  such that </text>
      <img src="https://render.githubusercontent.com/render/math?math=\Large \lambda_i \geq 0\:\forall\:i = i,2,.....,n">
        <text> and </text>
          <img src="https://render.githubusercontent.com/render/math?math=\Large {\sum_{i = 1}^n} {\lambda_i y_i} = 0 ">
</p>

Now, let us look at the terms individually.

![first term](https://github.com/Adi-ds/Private-Repo/blob/main/images/equatio.png)

<img src="https://equatio-api.texthelp.com/png/%5Cbegin%7Barray%7D%7Bl%7D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7B1%3D1%7D%5En%5Clambda_i%5Clambda_jy_iy_jx_i%5ETx_j%5C%5C%0D%0A%3D%5Cfrac%7B1%7D%7B2%7D%5Cleft(%5Clambda_%7B1%5C%20%7D%5Clambda_2%5C%20...%5C%20%5Clambda_n%5Cright)%5Cbegin%7Bpmatrix%7Dy_1y_1x_1%5ETx_1%26y_1y_2x_1%5ETx_2%26..........%26y_1y_nx_1%5ETx_n%26%5C%5C%0D%0Ay_2y_1x_2%5ETx_1%26y_2y_2x_2%5ETx_2%26..........%26y_2y_nx_2%5ETx_n%26%5C%5C%0D%0A..............%26..............%26..........%26..............%26%5C%5C%0D%0A..............%26..............%26..........%26..............%26%5C%5C%0D%0A..............%26..............%26..........%26..............%26%5C%5C%0D%0Ay_ny_1x_n%5ETx_1%26y_ny_2x_n%5ETx_2%26..........%26y_ny_nx_n%5ETx_n%26%5Cend%7Bpmatrix%7D%5Cbegin%7Bpmatrix%7D%5Clambda_1%5C%5C%0D%0A%5Clambda_2%5C%5C%0D%0A.%5C%5C%0D%0A.%5C%5C%0D%0A.%5C%5C%0D%0A%5Clambda_n%5C%5C%0D%0A%5Cend%7Bpmatrix%7D%5C%5C%0D%0A%3D%5Cfrac%7B1%7D%7B2%7D%5Cvec%7B%5Clambda%7D%5ETQ%5Cvec%7B%5Clambda%7D%5Cend%7Barray%7D?height=286">

<img src="https://equatio-api.texthelp.com/png/%5Csum_%7Bi%3D1%7D%5En%5Clambda_iy_i%5C%20%3D%5C%20%5Cleft(%5Clambda_1%5C%20%5Clambda_2%5C%20...%2C%5C%20%5Clambda_n%5Cright)%5Cleft%5B%5Cbegin%7Bmatrix%7Dy_1%5C%5C%0D%0Ay_2%5C%5C%0D%0A.%5C%5C%0D%0A.%5C%5C%0D%0A.%5C%5C%0D%0Ay_n%5Cend%7Bmatrix%7D%5Cright%5D%3D%5Cvec%7B%5Clambda%7D%5Cvec%7By%7D?height=194">

Now, we can write the dual problem as 

<img src = "https://equatio-api.texthelp.com/png/K%5C%20%3D%5C%20%5Cvec%7B%5Clambda%7D%5ET-%5C%20%5Cfrac%7B1%7D%7B2%7D%5Cvec%7B%5Clambda%7D%5ETQ%5Clambda%2B%5Cbeta%5Cvec%7B%5Clambda%7D%5ET%5Cvec%7By%7D?height=49"> where <img src="https://render.githubusercontent.com/render/math?math=\Large \beta"> is the Lagrange Multiplier.

So, <img src = "https://equatio-api.texthelp.com/png/%5Cbegin%7Barray%7D%7Bl%7D%5Cnabla_%7B%5Cvec%7B%5Clambda%7D%7DK%5C%20%3D%5C%20%5Cvec%7B1%7D-%5C%20Q%5Cvec%7B%5Clambda%7D%5C%20%2B%5C%20%5Cbeta%5Cvec%7By%7D%5C%20%3D%5C%200%5C%20%5CRightarrow%5C%20Q%5Cvec%7B%5Clambda%5C%20%7D%3D(%5Cvec%7B1%7D%5C%20%2B%5C%20%5Cbeta%5Cvec%7By%7D)%5C%20%5CRightarrow%5C%20%5Cvec%7B%5Clambda%7D%5E%7B*%7D%5C%20%3D%20%5C%20Q%5E%7B-1%7D(%5Cvec%7B1%7D%5C%20%2B%5C%20%5Cbeta%5Cvec%7By%7D)%5Cend%7Barray%7D?height=37">.

Here, <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{\lambda}^{*}"> is the obtained value of <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{\lambda}">.

Therefore, the obtained value of <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{w}"> will be <img src = "https://equatio-api.texthelp.com/png/w%5E*%3D%5C%20%5Csum_%7Bi%3D1%7D%5En%5Clambda_i%5E*y_ix_i?height=73">. 

Clearly, the optimal <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{w^{*}}"> is a function of <img src="https://render.githubusercontent.com/render/math?math=\Large \vec{\lambda}^{*}">.

Therefore, the optimal solution of <img src="https://render.githubusercontent.com/render/math?math=\Large b"> is given by <img src = "https://render.githubusercontent.com/render/math?math=\Large b^*\ =\ -\ \frac {\max_{i:y_i=-1}\vec{w}^*\vec{x_i}\ %2B \min_{i:y_i=1}\vec{w}^*\vec{x_i} }{2}">.
