import keras
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("./run/vae/1_output/model", compile=False)
layer = model.get_layer("model_2")
z_generated = np.random.normal(size=(1, 300))
result = layer.predict(np.array(z_generated))

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1):
    ax = fig.add_subplot(3, 10, i + 1)
    ax.imshow(result[i])
plt.show()
