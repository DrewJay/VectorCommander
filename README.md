# Plotz

This project includes configurable "vanilla" Variational Autoencoder.

## Model components
<b>Layers used:</b>
<ul>
    <li>Classical convolutional layers.</li>
    <li>Dropout layers (prevent overfitting).</li>
    <li>LeakyRelu activation layers.</li>
    <li>Flatten layer.</li>
</ul>

<b>Losses used:</b>
<ul><li>Composite loss of Kullback-Leibler divergence and MSE reconstruction.</li></ul>

## Model architecture
<img src="https://i.ibb.co/TMWjmMB/model.png"/>

## Demonstration
Following image simulates development of hernia on healthy lungs. Simulation is performed by finding unit vector representing
given feature (in this case hernia disorder), scaling by factorization and adding it back to original image.

<figure>
    <img src="https://i.ibb.co/NW3QCcb/factor.png"/>
    <figcaption>
        Fig.1 - Image factorization. Factor 0 is attempt of VAE to recreate original image. Factor f > 0 is image to
        which scaled hernia vector was added. Left lung is progressively expanding on bottom-left.
    </figcaption>
</figure>
<br/><br/>
<figure>
    <img src="https://i.ibb.co/MSTw23V/abs-diff.png"/>
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
VAEs are Adversarial Autoencoders, which don't use KL Divergence to measure quality of generated probability distribution,
but discriminator model being taught that generated PD is "fake" while standard normal distribution is "real". AAEs yield
better results by generating more organized latent space.
