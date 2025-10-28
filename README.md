# Fashion MNIST Classifier

A simple deep learning project that classifies clothing images (such as T-shirts, trousers, shoes, and bags) using the Fashion MNIST dataset and a TensorFlow Keras model. This project also includes a Streamlit web app for interactive predictions.

## Project Structure
```
Fashion-MNIST-Project/
│
├── app.py # Streamlit web app for predictions
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── saved_model/
│   └── fashion_mnist_model.h5 # Trained model file
│
└── src/
    ├── main.py # Entry point or utility runner
    ├── model.py # Model architecture definition
    ├── train.py # Model training script
    └── utils.py # Helper functions (class names, preprocessing, etc.)
```

## Model Summary
The classifier uses a simple fully connected neural network:

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

It’s trained on the Fashion MNIST dataset, which contains 70,000 grayscale images (10 clothing categories).

## Requirements
To install dependencies, create a virtual environment and install the requirements:

python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt

## Training the Model
To train or retrain your model, run:

python -m src.train

This will train the model on the Fashion MNIST dataset and save it in saved_model/fashion_mnist_model.h5.

## Run the Streamlit App
Once your model is trained and saved, launch the Streamlit app:

streamlit run app.py

You’ll see a web interface where you can upload 28×28 grayscale clothing images and view predictions.

## Example Output
Upload an image → The app predicts the clothing type (e.g., “Sneaker”, “T-shirt/top”)

Displays confidence → The model’s probability score for its prediction

## Dataset Info
The Fashion MNIST dataset includes the following 10 classes:

Label | Class Name
------|------------
0 | T-shirt/top
1 | Trouser
2 | Pullover
3 | Dress
4 | Coat
5 | Sandal
6 | Shirt
7 | Sneaker
8 | Bag
9 | Ankle boot
