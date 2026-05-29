from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
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
#plot_digit(some_digit)
#plt.show()

#print(y[0])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[0:60000], y[60000:]

# training a binary classifier

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

pred = sgd_clf.predict([some_digit])
print(pred)

cvs = cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring="accuracy")
print(cvs)