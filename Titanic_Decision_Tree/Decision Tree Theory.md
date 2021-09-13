# Decision Tree Theory

Decision Tree algorithm is a Supervised Machine Learning Algorithm. It works on both categorical and continuous input and output variables. So, the decision tree algorithm can be used for both classification and regression task. 

Decision tree is a flow chart like structure based on if-else conditions. In other words, a decision tree is a tree whose each interal node specifies a test on an attribute of the data and each edge between a parent and a child represents a decision based on that test.

As in the Titanic dataset, we have to classify between survived and not survived, so the classification algorithm will be used.

Now, decision tree algorithm works with mainly two impurity metrices :
1. Entropy
2. Gini Index/Gini Impurity

## Entropy

In layman terms, **Entropy** is nothing but the measure of homogeneity. The goal of Decision tree in general is to increase homogeneity. This metric allows the decision tree classifier in a way, such that, it ultimately attains maximum homogeniety possible in the leaf nodes.

Suppose, <img src="https://render.githubusercontent.com/render/math?math=\LARGE \red \mathcal{C}"> be the set of k classes, which are possible from a feature F of a dataset, i.e, <img src="https://render.githubusercontent.com/render/math?math=\LARGE \mathcal{C} = \{F_1, F_2, F_3, ..... , H_k\}">.

Now, let us define, <img src="https://render.githubusercontent.com/render/math?math=\LARGE p_i"> as the probability of occurence of <img src="https://render.githubusercontent.com/render/math?math=\LARGE F_i">, i.e,

<img src="https://render.githubusercontent.com/render/math?math=\LARGE p_i = \dfrac{number\:of\:occurrences\:of\:F_i}{total\:number\:of\:classes\:formed\:in\:F}">

The Entropy is defined by the formula : 

<img src="https://render.githubusercontent.com/render/math?math=\LARGE Entropy =  I(p_1, p_2, p_3,..., p_k) = -\sum_{i = 1}^k{p_i\,log\,p_i}"> 

Let us consider the following example :

In the EDA part we saw that, 

Survival Status | Count
:---------------|:------:
Survived        | 342
Died            | 549
Total           | 891

So here,

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  p_s = Probability\:of\:'Survived' = \dfrac {342}{891} = 0.38">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  p_d = Probability\:of\:'Died' = \dfrac {549}{891} = 0.62">

So,

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  Entropy(Survival) = -0.38 * log(0.38) - 0.62 * log(0.62) \approx 0.29">

The Entropy here is 0.29. This can be considered as a low entropy meaning a high level of homogeneity.

Entropy is measured between 0 and 1. The best classification is done when the resulting classes have low Entropy values.

### Information Gain

Information Gain is applied to quantify which feature provides maximal information about the classification based on the notion of entropy.

Let us consider the following information obtained from EDA.

Survival Status | Male | Female
:--------------:|:----:|:------:
Survived        | 109  | 233
Died            | 468  | 81
Total           | 577  | 314

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  P(Male|Survival Status = 'Survived') = \dfrac {109}{577} = 0.19">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  P(Male|Survival Status = 'Died') = \dfrac {468}{577} = 0.81">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  P(Female|Survival Status = 'Survived') = \dfrac {233}{314} = 0.74">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  P(Female|Survival Status = 'Died') = \dfrac {81}{314} = 0.002">

Information Gain is given by -

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  IG(Survival Status, Sex = 'Male')">
<img src="https://render.githubusercontent.com/render/math?math=\LARGE  = Entropy(Survival)">
<img src="https://render.githubusercontent.com/render/math?math=\LARGE  - Entropy(Survival Status = 'Survived'|Sex = 'Male')">
<img src="https://render.githubusercontent.com/render/math?math=\LARGE  - Entropy(Survival Status = 'Died'|Sex = 'Male')">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  = 0.29 - 0.19 * log(0.19) - 0.81 * log(0.81)">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  \approx 0.50">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  IG(Survival Status, Sex = 'Female')">
<img src="https://render.githubusercontent.com/render/math?math=\LARGE  = Entropy(Survival)">
<img src="https://render.githubusercontent.com/render/math?math=\LARGE  - Entropy(Survival Status = 'Survived'|Sex = 'Female')">
<img src="https://render.githubusercontent.com/render/math?math=\LARGE  - Entropy(Survival Status = 'Died'|Sex = 'Female')">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  = 0.29 - 0.74 * log(0.74) - 0.002 * log(0.002)">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  \approx 0.52">

So,

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  IG(Survival Status, Sex = 'Male') = 0.50">

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  IG(Survival Status, Sex = 'Female') = 0.52">

## Gini Index

Gini index of value 0 means sample are perfectly homogeneous and all element are similar, whereas, Gini index of value 1 means maximal inequality among elements. It is sum of the square of the probabilities of each class.

Again let us consider that, 

<img src="https://render.githubusercontent.com/render/math?math=\LARGE  \mathcal{C}"> be the set of k classes, which are possible from a feature F of a dataset, i.e, <img src="https://render.githubusercontent.com/render/math?math=\LARGE \mathcal{C} = \{F_1, F_2, F_3, ..... , H_k\}">.

Now, let us define, <img src="https://render.githubusercontent.com/render/math?math=\LARGE p_i"> as the probability of occurence of <img src="https://render.githubusercontent.com/render/math?math=\LARGE F_i">, i.e, 

<img src="https://render.githubusercontent.com/render/math?math=\LARGE p_i = \dfrac{number\:of\:occurrences\:of\:F_i}{total\:number\:of\:classes\:formed\:in\:F}">

The Gini Index is defined by the formula :

<img src="https://render.githubusercontent.com/render/math?math=\LARGE Gini Index = \sum_{i = 1}^k{p_i}{(1-p_i)} = 1-\sum_{i = 1}^k{p_i}^2">


NB : Information Gain uses Entropy as the base calculation ad so it has a wider range of results whereas the Gini Index caps at one. The Gini Index facilitates the bigger distributions so easy to implement whereas the Information Gain favors lesser distributions having small count with multiple specific values. Gini index operates on the categorical target variables in terms of “success” or “failure” and performs only binary split, in opposite to that Information Gain computes the difference between entropy before and after the split and indicates the impurity in classes of elements.
