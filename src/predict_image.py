import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Class labels
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
          'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load model and data
model = load_model("../model/cifar10_cnn_model.h5")
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

# Predict
index = 8
img = x_test[index]
prediction = model.predict(np.expand_dims(img, axis=0))
predicted_label = np.argmax(prediction)
actual_label = y_test[index][0]

# Show image
plt.imshow(img)
plt.title(f"Predicted: {labels[predicted_label]} | Actual: {labels[actual_label]}")
plt.axis('off')
plt.savefig("../images/output.png")
plt.show()
