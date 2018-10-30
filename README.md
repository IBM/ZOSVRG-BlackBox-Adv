# ZOSVRG for Generating Universal Attacks on Black-box Neural Networks

ZOSVRG is the proposed new zeroth-order nonconvex optimization method. This repo presents ZOSVRG's application for generating adversarial attacks on black-box neural networks. It contains a pretrained network model for the MNIST classification task, and a Python implementation for attack generation that can directly be applied to the network model.

For the ZOSVRG algorithm, see our NIPS 2018 paper “[Zeroth-Order Stochastic Variance Reduction for Nonconvex Optimization](https://arxiv.org/abs/1805.10367)” (Hereinafter referred to as Paper.)


## Description
This Python code generates universal adversarial attacks on neural networks for the MNIST classification task under the black-box setting. For an image **x**, the universal attack **d** is first applied to **x** in the *arctanh* space. The final adversarial image is then obtained by applying the *tanh* transform. Summarizing, **x**<sub>adv</sub> = *tanh*(*arctanh*(2**x**) + **d**)/2

Below is a list of parameters that the present code takes:
1. -optimizer: This parameter specifies the optimizer to use during attack generation. Currently the code supports ZOSGD and ZOSVRG.
2. -q: The number of random vector to average over when estimating the gradient.
3. -alpha: The optimizer's step size for updating solutions is alpha/(dimension of **x**)

## Example 1
python3 Universal_Attack.py -optimizer ZOSVRG -q 10 -alpha 1.0 -M 10 -nStage 25000 -const 1 -nFunc 10 -batch_size 5 -mu 0.01 -target_label 4 -rv_dist UnitSphere

<img src="/Sample-Output/ZOSVRG-Sample-1/0004.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0006.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0019.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0024.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0027.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0033.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0042.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0048.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0049.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/0056.png" width="80" height="80">

<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id4_Orig4_Adv9.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id6_Orig4_Adv8.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id19_Orig4_Adv2.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id24_Orig4_Adv9.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id27_Orig4_Adv9.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id33_Orig4_Adv2.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id42_Orig4_Adv9.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id48_Orig4_Adv9.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id49_Orig4_Adv9.png" width="80" height="80"><img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id56_Orig4_Adv9.png" width="80" height="80">


<!---
-M: help="Length of each stage/epoch")
-nStage: help="Number of stages/epochs")
-const: help="Weight put on the attack loss")
-nFunc: help="Number of images being attacked at once")
-batch_size: help="Number of functions sampled for each iteration in the optmization steps")
-mu: help="The weighting magnitude for the random vector applied to estimate gradients in ZOSVRG")
-target_label: help="The target digit to attack")
-->

# Updates Upcmoing Soon
