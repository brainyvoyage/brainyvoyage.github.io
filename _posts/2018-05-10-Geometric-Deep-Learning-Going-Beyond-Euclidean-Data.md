---
layout:     post
title:      Geometry of Manifold and Graph
date:       2018-05-10 11:21:29
summary:    Geometric DL- Umbrella term for emerging techniques attempting to generalise structured deep neural networks to non-Euclidean domains such as graph and manifold
categories: Geometric-DL Deep-Learning
use_math:   true
# exclude page from index
meta_robots: noindex
---
[Bronstein et. al](https://arxiv.org/pdf/1611.08097.pdf) *IEEE Signal Processing*

# Introduction
**Geometric DL**: Umbrella term for emerging techniques attempting to generalise (structured) deep neural networks to non-Euclidean domains such as graph and manifold


## Two classes of geometric learning problems

1. Characterisation of the structure of the data
2. Analysing functions defined on a given non-euclidean domain

### Structure of the domain

* Lower dimensional structure embedded into a high-dimensional Euclidean space
* Manifold Learning: Recovering lower dimensional structure
* Manifold learning is also referred as non-linear dimensionality reduction. Involves
  * construction of lower affinity structure of the given data points (graph)
  * Embedding of the data points into low-dimensional space with some criteria to preserve original affinity
  * Example: Multidimensional Scaling (MDS), locally linear embdedding (LLE), t-SNE, Spectral embedding such as Laplacian eigenmaps and diffusion maps
* In some cases, data are presented as graph or manifold and construction of the affinity structure is unnecessary
  * graph: co-occurrences of words in NLP
  * Manifold: Computer graphics and vision applications

### Data on a Domain
*Preliminary*

$L^2$-function:
A function  $f(x):(-\infty, \infty)\rightarrow \mathbb{R}$. The function $f$ is $L^2$ if $\int_{-\infty}^\infty f(x)^2 dx \lt \infty$
Analysing functions defined on a given non-Euclidean domain.

Two subclasses:

* Fixed domain [Social graph]
  * [Methods of signal processing ](https://arxiv.org/pdf/1211.0053.pdf)on graph can be applied in order to define an operation similar to convolution in spectral domain as in [[PDF](https://arxiv.org/pdf/1506.05163.pdf)] [[PDF](https://arxiv.org/pdf/1606.09375.pdf)]
* Multiple domain: Finding similarity and correspondence between shapes [Computer Graphics and Vision applications]

## Deep Learning on Euclidean Domains

### Geometric Priors
#### Stationarity
*Translation Operator*

Let $$\begin{align}\mathcal{T}_v f(x) = f (x-v) \end{align}$$

be a translation operator where $\Omega = [0, 1]^d \subset \mathbb{R}^d $ d-dimensional Euclidean domain on which function $f \in L^2(\Omega)$ is defined,
x, v $\in \Omega$. In most of the vision and speech task there are prior $\textbf{assumptions}$ on unknown function $y: L^2(\Omega) \rightarrow \mathcal{Y}$ 
[supervised learning]

The *first assumption* is that the function *y* is 
1.  invariant [typically in classification task] or
2.  equivariant [object localisation, semantic segmentation, motion estimation]

With respect to translations
$$
\begin{align}
\text{Invaraint: } y\big(\mathcal{T}_v f\big) &= y(f)\\
\text{Equivariant: }y\big(\mathcal{T}_v f\big) &= \mathcal{T}_vy(f)
\end{align}
$$
#### Local deformation and scale separation:
Deformation can model local translations, changes in point of view, rotations and frequencey transposition. Vision tasks are also stable against deformations 

Let deformation $\mathcal{L}_t$  where $\tau : \Omega \rightarrow \Omega$ be a smmoth vector field acts as 

$$
\begin{align}
\mathcal{L}_{\tau}f(x) &= f(x - \tau(x))
\end{align}
$$
In case of translation invariant we have
$$
\begin{align}
\big\vert y\big(\mathcal{L}_{\tau}f\big)\big\vert \approx ||\nabla_\tau||
\end{align}
$$

Here 
$
\vert\vert{\nabla}_{\tau}\vert\vert
$ is measeure of smoothness

> From above equation, it follows that we can extract sufficient statistics at a lower spatial resolution by downsampling demodulated localised filter without losing approximation power.

**Convolution Neural Network** leverages both (a) *Stationarity* (b) *Stability* to local transformation