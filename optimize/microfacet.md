# A few must-read BRDF papers

[Back to Main Page](../README.md)

## Microfacet Models for Refaction through Rough Surface
[Paper](https://hal.science/hal-04001287v1/file/MIPNet.pdf). It is the paper that intorduces GGX after laying out properties of a physically correct normal distribution.

## Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs
[Paper](https://jcgt.org/published/0003/02/03/paper.pdf). Another paper that discuss the fundmental properties of a physicall correct micro-facet material model. It also introduces two masking functions for anisotropic Beckman and GGX distribution. The paper comes along a supplementary [slides deck](https://jcgt.org/published/0003/02/03/presentation.pdf) which is more readable than paper itself. 

<img src="images/mask.png" alt="MIPNet neural architecture" style="display: block; margin: 0 auto;">

## The SGGX Microflake Distribution
[Paper](https://drive.google.com/file/d/0BzvWIdpUpRx_dXJIMk9rdEdrd00/view?usp=sharing&resourcekey=0-ZS9wFi1rJvENbyWTH6BAFA). [Supplementary](https://drive.google.com/file/d/0BzvWIdpUpRx_djVyMG9jMnltdTg/view?usp=sharing&resourcekey=0-VTvjBPesVjrNy4SH2ShqDw) This paper represents the normal distribution with a position symmetric 3x3 matrix, namely, $D(\omega_m)=\sqrt{\omega_m^T S \omega_m}$. It is surprising that a NDF can be represented in such a simple and elegant way. The essential observation is 

1. If we project the microfacets along with a direction, the projected area is $\sigma(v) = (v, n) = \int D(\omega_m)(v \cdot \omega_m)d \omega_w$, where $n$ is the geometric normal.
2. This projected area can be also represented with $\sqrt{v^T S v}$

