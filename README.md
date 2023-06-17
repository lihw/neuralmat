# Neural Material Algorithms Survey

An extensive review of neural material algorithms that have emerged in academic publications and startups over the past few years, including but not limited to neural material representation, importance sampling about neural materials, material compression, mipmap generation using neural materials and so.

## Table of Contents

- [Introduction](#introduction)
- [Algorithms](#algorithms) [Optimizing BRDFs](#optimizing-brdfs) [Neural BRDFs](#neural-brdfs)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Neural material has been a popular research area in recent years, with two major research directions. The first involves using neural networks to represent BRDF and incorporating the new representation into the renderer. The second direction focuses on optimizing the current BRDF data, e.g., BRDF compression and authoring workflow. This repository aims to gather all relevant papers, code and tools that can assit researchers in reproducing the results or creating their own proof-of-concepts. Additionally, we may provide a all-in-one GUI tool in the future to consolidate all these resources like NerfStudio.

## Algorithms

As previously mentioned, we have categorized all recent neural material advances into two groups. 

## Optimizing BRDFs

- **Create mipmaps with neural network or gradient descent methods**. The goal is to improve the quality of normal, roughness, and height maps by creating mipmaps using neural network or gradient descent methods. This is a challenge because these signals cannot be easily linearly interpolated or will be transformed to anisotropic ones in another domain. For instance, the linear sum of two opposite normal vectors is zero, and the downsampled height maps may show anisotropic roughness. This approach aims to generate better results than using a simple Gaussian filter. Refer to [mipmap.md](optimize/mipmap.md) for algrithms addressing this problem.

- **Compress SVBRDFs with neural network**. By utilizing the similarities between different channels of the SVBRDF textures, e.g., between roughness and albedo, we can further compress SVBRDFs down to a new level. 

## Neural BRDFs


## Installation

For instructions on installation of each algorithm, please refer to their respective documentation.

## Contributing

Explain how others can contribute to your project. For example:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - lihw81@gmail.com

Project Link: https://github.com/lihw/neuralmat.git