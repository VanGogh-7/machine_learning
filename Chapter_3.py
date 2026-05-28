from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame = False)

X, y = mnist.data, mnist.target

#print(X, y)
#print(X.shape)
#print(y.shape)
def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap = 'binary')
    plt.axis('off')

some_digit = X[0]
plot_digit(some_digit)
plt.show()

#print(y[0])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[0:60000], y[60000:]
