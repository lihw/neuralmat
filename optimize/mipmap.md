# Create mipmaps with neural network or gradient descent methods

[Back to Main Page](../README.md)

## MIPNet:Neural Normal-to-Anisotropic-Roughness MIP mapping
This paper dicusses a neural method for creating mipmap of SVBRDFs and asserts it can outperform traditional gaussian blur, [LEAN](https://redirect.cs.umbc.edu/~olano/papers/lean/) and [LEADR](https://inria.hal.science/hal-00858220v1/document) filters in most scenairos. 

### [Project page](https://perso.telecom-paristech.fr/boubek/papers/MIPNet/) | [Paper](https://dl.acm.org/doi/pdf/10.1145/3550454.3555487) | [Presentation](https://www.youtube.com/watch?v=Pij9z3auXsc) | [Code](https://github.com/AlbanGauthier/mipnet_neural_mipmap)

This paper proposes a cascaded architecture of multilayer perceptron architecture. This architecture is more robust in error accumulation compared to a single-resolution neural architecture, which uses one MLP to downsample each level of mipmap. The network architecture is shown below,

<img src="images/mipnet.PNG" alt="MIPNet neural architecture" style="display: block; margin: 0 auto;">

### Keypoints
- **The NN model which composes of two networks, H_A and H_B seems complex and unwieldly**

    Firstly, if we were to use H_A alone (i.e., just one network), we would not be able to feed it with both LoD_i and LoD_{i-1} at the same time. This is because there are no two inputs at the first level of mipmap.

    Secondly, if we were to use H_A only at the first level and H_B for the rest (two networks but use them at different levels), H_A would not learn any anisotropy signals that only appear during downsampling. The anistropy are indeed the the anisotropic distribution of normals and roughness given a large pixel footprint. The topmost level (the input image) is isotropic per pixel.

- **Why does simply downsampling not work?**

    It works for albedo, but it doesn't work for normal and roughness maps. When rendering object at afar, the footprint of a pixel is large and covers many texels. Therefore, it is neccessary to render at a subpixel level, where each subpixel corresponds to one texel, and average/filter the subpixel shading results. This process is known as "downsampling" the shading result. However, simply downsampling normal and roughness maps does not work because the shading reuslt is not equal to the former one. This is because normal and roughness maps cannot be easily linearly interpolated or transformed to anisotropic ones in another domain. (albedo works because for non-metallic materials, they are a scalar multiplier to the shading equation and can be move outside of the integration). This problem has been addressed in [LEAN](https://redirect.cs.umbc.edu/~olano/papers/lean/) and [LEADR](https://inria.hal.science/hal-00858220v1/document).

- **The tensor representation of anisotropic roughness**

    If we use oridinary representation of $(\alpha_{t}, \alpha_{b})$, it is ambiguity when the anisotropic space $(t, b, n)$ rotates $180$ degrees. That is to say the shading equation will give the same result with opposite $(t, b, n)$ vectors.
    $$GGX = \frac{\langle h \cdot n \rangle}{\pi \alpha_{t}\alpha_{b}(\frac{(t \cdot h)^2}{\alpha_{a}^2}+\frac{(b \cdot h)^2}{\alpha_{b}^2}+(n \cdot h))^2} $$
    When we transfrom the equation to tensor representation, it becomes deterministic. 
    $$ \frac{(t \cdot h)^2}{\alpha_{a}^2}+\frac{(b \cdot h)^2}{\alpha_{b}^2} =  (\frac{(t \cdot h)}{\alpha_{a}},\frac{(b \cdot h)}{\alpha_{b}}) \cdot (\frac{(t \cdot h)}{\alpha_{a}},\frac{(b \cdot h)}{\alpha_{b}})^T = (A B h)^T \cdot (A B h) $$

    , where $A = R\begin{pmatrix}
\alpha_{t} & 0 \\
0 & \alpha_{b}
\end{pmatrix} R^T$, $R$ is a rotation matrix that transforms the normal to $(0, 0, 1)$ in anisotropic space and B is $(t, b)^T$.

    The full tensore presentation can be found in [SGGX supplementary](https://drive.google.com/file/d/0BzvWIdpUpRx_djVyMG9jMnltdTg/view?resourcekey=0-VTvjBPesVjrNy4SH2ShqDw), where GGX is rewritten to 
    $$\frac{\langle h \cdot n \rangle}{\pi\sqrt{\lvert S\rvert}(h^t S h)^2}$$
    , where $S=B\begin{pmatrix}
    \alpha_{t}^2 & 0 & 0 \\
0 & \alpha_{b}^2 & 0 \\
0 & 0 & 1
\end{pmatrix}B^T$, and $B = (t, b, n)^T$ now. 

- **The network pseudo code**
    ```python
    class ModelA(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.seq = []

            self.conv2 = nn.Conv2d(2 + 3 + 1, 512, kernel_size = (2, 2), stride = (2, 2))) # (nx, ny, a, b, c, h)
            self.fc1   = nn.Linear(512, 512))
            self.fc2   = nn.Linear(512, 512))
            self.fc3   = nn.Linear(512, 2 + 3))

        def forward(self, x):
            half = torch.mean(x, dim = 1)
            x = nn.ReLU(self.conv2(x))
            x = nn.ReLU(self.fc1(x))
            x = nn.ReLU(self.fc2(x))
            x = self.fc3(x) +

            return x

    class ModelB(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.seq = []

            self.conv1 = nn.Conv2d(2 + 3 + 1, 1024, kernel_size = (1, 1), stride = (1, 1))) # (nx, ny, a, b, c, h)
            self.conv2 = nn.Conv2d(2 + 3 + 1, 1024, kernel_size = (2, 2), stride = (2, 2))) 
            self.conv4 = nn.Conv2d(2 + 3 + 1, 1024, kernel_size = (4, 4), stride = (4, 4))) 
            
            self.fc1   = nn.Linear(1024, 1024))
            self.fc2   = nn.Linear(1024, 1024))
            self.fc3   = nn.Linear(1024, 1024))
            self.fc4   = nn.Linear(1024, 2 + 3))
    
        def forward(self, lod0, lod1, lod2): # lod0 and lod2 are two upper levels, and lod1 is the output of modelA
            x = nn.ReLU(self.conv1(x) + self.conv2(x) + self.conv4(x))
            x = nn.ReLU(self.fc1(x))
            x = nn.ReLU(self.fc2(x))
            x = nn.ReLU(self.fc3(x))
            x = nn.ReLU(self.fc4(x))

            return x

    ```

### Troubleshoots
There may be a few problems to run the project code.

- `ImportError: DLL load failed while importing cv2: The specified module could not be found.`
It looks the installation section in mipnet has missed to add `pip install opencv-contrib-python`.

- `ModuleNotFoundError: No module named 'tensorboard'`
Simply run `pip install tensorboard`.

### Results

### 
