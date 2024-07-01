# Lung Cancer Prediction using CNN and Transfer Learning

This project aims to build a Lung Cancer Prediction System using Convolutional Neural Networks (CNN) and transfer learning. The model classifies lung cancer images into four categories: Normal, Adenocarcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma.


## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Using the Model](#using-the-model)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection and accurate classification are crucial for effective treatment and patient survival. This project leverages deep learning techniques to develop a robust lung cancer classification model using chest X-ray images.

## Dataset

The dataset used in this project consists of lung cancer images categorized into four classes:
1. Normal
2. Adenocarcinoma
3. Large Cell Carcinoma
4. Squamous Cell Carcinoma

The dataset should be organized into training (`train`), validation (`valid`), and testing (`test`) folders with the following subfolders for each class:

- `train/`
  - `normal/`
  - `adenocarcinoma/`
  - `large_cell_carcinoma/`
  - `squamous_cell_carcinoma/`

- `valid/`
  - `normal/`
  - `adenocarcinoma/`
  - `large_cell_carcinoma/`
  - `squamous_cell_carcinoma/`

- `test/`
  - `normal/`
  - `adenocarcinoma/`
  - `large_cell_carcinoma/`
  - `squamous_cell_carcinoma/`

Alternatively, you can also download a similar dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) which includes Chest CT scan images.

### Google Colab Link
To replicate and run the project in Google Colab, use the following link: [Lung Cancer Prediction System on Colab](https://colab.research.google.com/drive/1kMTghEwVoJaFmlKydxuhhoyzHluIUjoV?usp=sharing)


### Usage

- **Direct Download**: You can download the dataset directly from this repository and store it on your local system.
- **Google Drive**: Alternatively, you can store the dataset in your Google Drive and mount it using the provided code to replicate the environment used in this project.

## Dependencies

The project requires the following libraries:
- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- tensorflow
- keras

You can install the required libraries using the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow keras
```


## Project Structure

```
.
├── Lung_Cancer_Prediction.ipynb
├── README.md
├── dataset
│ ├── train
│ │ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ │ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ │ ├── normal
│ │ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
│ ├── test
│ │ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ │ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ │ ├── normal
│ │ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
│ └── valid
│ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ ├── normal
│ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
└── best_model.hdf5
```

This structure outlines the files and directories included in your project:

- **Lung_Cancer_Prediction.ipynb**: Jupyter Notebook containing the code for training and evaluating the lung cancer prediction model.
- **README.md**: Markdown file providing an overview of the project, usage instructions, and other relevant information.
- **dataset/**: Directory containing the dataset used for training and testing.
  - **train/**: Subdirectory containing training images categorized into different classes of lung cancer.
  - **test/**: Subdirectory containing testing images categorized similarly to the training set.
  - **valid/**: Subdirectory containing validation images categorized similarly to the training set.
- **best_model.hdf5**: File where the best-trained model weights are saved after training.




## Training the Model

The Jupyter Notebook `Lung_Cancer_Prediction.ipynb` contains the code for training the model. Below are the steps involved:

1. **Mount Google Drive**: To access the dataset stored in Google Drive.
2. **Load and Preprocess Data**: Use `ImageDataGenerator` for data augmentation and normalization.
3. **Define the Model**: Use the Xception model pre-trained on ImageNet as the base model and add custom layers on top.
4. **Compile the Model**: Specify the optimizer, loss function, and metrics.
5. **Train the Model**: Fit the model on the training data and validate it on the validation data. Callbacks like learning rate reduction, early stopping, and model checkpointing are used.
6. **Save the Model**: Save the trained model for future use.

### Example Usage

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Load and preprocess data
IMAGE_SIZE = (350, 350)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=IMAGE_SIZE,
    batch_size=8,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validate_folder,
    target_size=IMAGE_SIZE,
    batch_size=8,
    class_mode='categorical'
)

# Define the model
pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
pretrained_model.trainable = False

model = Sequential([
    pretrained_model,
    GlobalAveragePooling2D(),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=25,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=20
)

# Save the model
model.save('/content/drive/MyDrive/dataset/trained_lung_cancer_model.h5')
```



## Using the Model

To use the trained model for predictions, follow these steps:

1. **Load the Trained Model**: Load the saved `.h5` model file using TensorFlow/Keras.
2. **Preprocess the Input Image**: Load and preprocess the input image using `image.load_img()` and `image.img_to_array()`.
3. **Make Predictions**: Use the loaded model to predict the class of the input image.
4. **Display Results**: Display the input image along with the predicted class label.

### Example Code

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('/content/drive/MyDrive/dataset/trained_lung_cancer_model.h5')

def load_and_preprocess_image(img_path, target_size):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

# Example usage with an image path
img_path = '/content/test_image.png'
target_size = (350, 350)

# Load and preprocess the image
img = load_and_preprocess_image(img_path, target_size)

# Make predictions
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])

# Map the predicted class to the class label
class_labels = list(train_generator.class_indices.keys())  # Assuming `train_generator` is defined
predicted_label = class_labels[predicted_class]

# Print the predicted class
print(f"The image belongs to class: {predicted_label}")

# Display the image with the predicted class
plt.imshow(image.load_img(img_path, target_size=target_size))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
```



## Results

After training and evaluating the lung cancer prediction model, the following results were obtained:

- Final training accuracy: `history.history['accuracy'][-1]`
- Final validation accuracy: `history.history['val_accuracy'][-1]`
- Model accuracy: 93%


### Example Predictions

Include images and their predicted classes here, demonstrating the model's performance on new data.




## Acknowledgements

We acknowledge and thank the contributors to the [Chest CT Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) on Kaggle for providing the dataset used in this project.




## License

This project is licensed under the [MIT License](LICENSE).

Feel free to use, modify, or distribute this code for educational and non-commercial purposes. Refer to the LICENSE file for more details.





