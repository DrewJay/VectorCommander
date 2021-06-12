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
Following image simulates development of hernia on healthy lungs. Pixelated character of the image can be improved by
increasing amount of filters in encoder convolutional layers (currently fixed at 64), or using alternative to MSE
reconstruction loss, for example pretrained image recognition model.

<img src="https://i.ibb.co/hHVHTyh/hernia.png"/>
