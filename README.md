<p align="center">
    <img width="300" src="https://i.ibb.co/7vR32Bw/vclogo2.png"/>
</p>

# VectorCommander

This project includes configurable Variational Autoencoder which can be customized and trained. It is implemented using modern approaches, like
discriminative regularization, <a href="https://arxiv.org/abs/1512.03385">residual convolutional blocks</a>, <a href="https://arxiv.org/pdf/1804.03599.pdf">gamma Lagrange multiplier with capacity.</a> It also uses experimental techniques like delta factorization of Gaussian Multivariate.

Part of the project is also file src/analysis.py which contains code for utilization of trained model and
methods performing vector operations over latent spaces.

## Model components
<b>Layers used:</b>
<ul>
    <li>Classical convolutional layers</li>
    <li>Dropout layers (prevent overfitting)</li>
    <li>LeakyRelu activation layers</li>
    <li>Flatten layers</li>
    <li>BatchNormalization layers</li>
    <li>Custom layers (generating Multivariate Gaussian)</li>
</ul>

<b>Losses used:</b>
<ul>
    <li>Composite loss of Kullback-Leibler divergence and MSE reconstruction</li>
    <li>Dynamic discriminative loss</li>
</ul>

## Demonstration
Following image simulates development of hernia on healthy lungs. Simulation is performed by finding unit vector representing
given feature (in this case hernia disorder), scaling by factorization and adding it back to original image.

<figure>
    <img src="https://i.ibb.co/gt7H9kh/factor.png"/>
    <figcaption>
        Fig.1 - Image factorization. Factor 0 is attempt of VAE to recreate original image. Factor f > 0 is image to
        which scaled hernia vector was added. Left lung is progressively expanding on bottom-left.
    </figcaption>
</figure>
<br/><br/>
<figure>
    <img src="https://i.ibb.co/QQBSb1N/abs-diff.png"/>
    <figcaption>
        Fig.2 - Visualization of how results change with increased factorization. Last image represents difference
        between factor 0 and 2. Left lung bottom-left expansion is confirmed.
    </figcaption>
</figure>
<br/><br/>

### Side note
Pixelated character of the image can be improved by
increasing amount of filters in encoder convolutional layers (currently fixed at 64), increasing latent vector size,
or using alternative to MSE
reconstruction loss, for example pretrained image recognition model.

Another better performing alternative to classical
VAEs are Adversarial AutoEncoders, which don't use KL Divergence to measure quality of generated probability distribution,
but discriminator model being taught that generated PD is "fake" while standard normal distribution is "real". AAEs yield
better results by generating more organized latent space.

## Usage
You can train your model by choosing your own configuration in src/settings/constants.py
and then running src/main.py.

File src/analysis.py contains implementations that utilize trained model and generate
visual results using matplotlib. It accepts following arguments:

Argument | Description | Type | Default value |
--- | --- | --- | --- |
vector_transition | Visualize continuous vector transition. | positional | N/A |
vector_lookup | Find and save vectors as numpy arrays. | positional | N/A |
vector_reconstruction | Reconstruct randomly selected images from latent space. | positional | N/A
random_samples | Generate random samples from latent space. | positional | N/A
column | CSV column name to seek label in. | str | N/A |
label | Value of the column. | str | N/A |
neutral_label | Label that doesn't include target state. | str | "No Finding" |
f_target | Target value of factorization. | int | 5 |
f_steps | Total amount of factorization steps between 0 and {f_target}. | int | 6 |
samples | Amount of samples to display. | int | 1 |

This line was used to generate results in Demonstration section:
```
py src/analysis.py --vector_transition --column "Finding Labels" --label "Hernia" --f_target 2 --f_steps 10 --samples 2
```
