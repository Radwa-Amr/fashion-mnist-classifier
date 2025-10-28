from tensorflow import keras
from src.model import build_model

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = build_model()

# تدريب لمدة كافية
model.fit(x_train, y_train, epochs=15, validation_split=0.1)

# تقييم
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# حفظ النموذج
model.save("saved_model/fashion_mnist_model.h5")
print("💾 Model saved successfully!")
