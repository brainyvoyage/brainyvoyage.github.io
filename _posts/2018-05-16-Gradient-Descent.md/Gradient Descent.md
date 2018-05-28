---
layout:     post
title:      Gradient Descent
date:       2018-05-16 11:54:29
summary:    Most of the cost function cannot to solved in close-form. Gradient descent is a first-order iterative optimisation algorithm to find minima/maxima(ascent) of a function.
categories: Deep-Learning Machine-Learning Optimization
use_math:   true
comments:   true
# exclude page from index
meta_robots: noindex
---
<centre>
<iframe width="560" height="315" src="https://www.youtube.com/embed/rhVIF-nigrY?rel=0&amp;showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
</centre>
## Gradient Calculation
In the last few videos, we learned that in order to minimize the error function, we need to take some derivatives. So let's get our hands dirty and actually compute the derivative of the error function. The first thing to notice is that the sigmoid function has a really nice derivative. Namely,

$$
\begin{align}
\sigma'(x) = \sigma(x)(1-\sigma(x))
\end{align}
$$

The reason for this is the following, we can calculate it using the quotient formula:
$$
\begin{align}
\sigma'(x) &= \frac{\partial}{\partial x}\frac{1}{1+e^{-x}}\\
&= \frac{e^{-x}}{(1+e^{-x})^2} \\
&= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}}\\
&= \sigma(x)(1-\sigma(x))
\end{align}
$$

And now, let's recall that if we have `m` points labelled $x^{(1)}, x^{(2)}, \dots, x^{(m)}$ the formula is:
$$
\begin{align}
E = -\frac{1}{m}\sum_{i=1}^m\big(y_i ln(\hat{y_i}) + (1 - y_i) ln(1 -\hat{y_i})\big)
\end{align}
$$

where prediction is given by $\hat{y_i} = \sigma(Wx^{(i)} + b)$
Our goal is to calculate the gradient of $E$, at a point $x = (x_1, x_2, \dots, x_n)$, given the partial derivatives
$$
\begin{align}
\nabla E = \bigg(\frac{\partial}{\partial{w_1}}E, \cdots, \frac{\partial}{\partial{w_n}}E, \frac{\partial}{\partial{b}}E\bigg)
\end{align}
$$

To simplify our calculations, we'll actually think of the error that each point produces, and calculate the derivative of this error. The total error, then, is the average of the errors at all the points. The error produced by each point is, simply,

$$
\begin{align}
E = - y_i ln(\hat{y}) - (1 - y) ln(1 -\hat{y})
\end{align}
$$

In order to calculate the derivative of this error with respect to the weights, we'll first calculate $\frac{\partial}{\partial w_j}\hat{y}$

$$
\begin{align}
\frac{\partial}{\partial w_j}\hat{y} &= \frac{\partial}{\partial w_j}\sigma(Wx^{(i)} + b)\\
&= \sigma\big(Wx + b\big)\big(1-\sigma(Wx + b)\big) \cdot \frac{\partial}{\partial w_j}\big(Wx+b\big) \\
&= \hat{y}(1-\hat{y})\cdot \frac{\partial}{\partial w_j}\big(Wx+b\big) \\
&= \hat{y}(1-\hat{y})\cdot \frac{\partial}{\partial w_j}\big(w_1x_1 + \dots + w_jx_j + \dots + w_nx_n +b\big)\\
&= \hat{y}(1-\hat{y})\cdot x_j
\end{align}
$$

The last equality is because the only term in the sum which is not a constant with respect $w_j$ is precisely $w_jx_j$ which clearly has deivative $x_j$

Now, we can go ahead and calculate the derivative of the error $E$ at a point $x$, with respect to the weoght $w_j$

$$
\begin{align}
\frac{\partial}{\partial w_j}E &= \frac{\partial}{\partial w_j}\bigg[-ylog(\hat{y}) - \big(1 - y\big)log\big(1 - \hat{y}\big)\bigg]\\
&= -y\frac{\partial}{\partial w_j}log(\hat{y}) - \big(1 - y\big)\frac{\partial}{\partial w_j}log\big(1 - \hat{y}\big)\\
&= -y\cdot\frac{1}{y}\frac{\partial}{\partial w_j}\hat{y} - \big(1 - y\big)\frac{1}{1-\hat{y}}\frac{\partial}{\partial w_j}\big(1 - \hat{y}\big)\\
&= -y\cdot\frac{1}{y}\hat{y}\big(1-\hat{y}\big)x_j - \big(1 - y\big)\cdot\frac{1}{1 - \hat{y}}\big(-1\big) \hat{y}\big(1-\hat{y}\big)x_j\\
&= -y\big(1-\hat{y}\big)\cdot x_j + \big(1 - y\big)\hat{y} \cdot x_j\\
&= -(y - \hat{y})x_j
\end{align}
$$
Similarly,
$$
\begin{align}
\frac{\partial}{\partial b} E = -(y - \hat{y})
\end{align}
$$
This actually tells us something very important. For a point with coordinates $(x_1, \cdots, x_n)$, label $y$, and prediction $\hat{y}$, the gradient of the error function at that point is
$$
\begin{align}
\big(-(y - \hat{y})x_1, \cdots, -(y - \hat{y})x_n, -(y - \hat{y})\big)
\end{align}
$$
In summary, the gradient is
$$
\begin{align}
\nabla E = -(y - \hat{y})(x_1, \cdots, x_n, 1)
\end{align}
$$
If you think about it, this is fascinating. The gradient is actually a scalar times the coordinates of the point! And what is the scalar? Nothing less than a multiple of the difference between the label and the prediction. What significance does this have?
* Closer the label to the prediction, smaller the gradient.
* Farther the lable from the prediction, larger the gradient

So, a small gradient means we'll change our coordinates by a little bit, and a large gradient means we'll change our coordinates by a lot.

If this sounds anything like the perceptron algorithm, this is no coincidence! We'll see it in a bit.

## Gradient Descent Step
Therefore, since the gradient descent step simply consists in subtracting a multiple of the gradient of the error function at every point, then this updates the weights in the following way:

$$
\begin{align}
w'_i \leftarrow w_i - \alpha[-(y-\hat{y})x_i],
\end{align}
$$
which is equivalent to
$$
\begin{align}
w'_i \leftarrow w_i + \alpha(y-\hat{y})x_i,
\end{align}
$$
Similarly, it updates the bias in the following woay:
$$
\begin{align}
b' \leftarrow b + \alpha(y-\hat{y})
\end{align}
$$
>Note: Since we've taken the average of the errors, the term we are adding should be $\frac{1}{m}\cdot \alpha$ instead of $\alpha$, but as $\alpha$ is a constant, then in order to simplify calculations, we'll just take $\frac{1}{m}\cdot\alpha$ to be our learning rate, and abuse the notation by just calling it $\alpha$.
