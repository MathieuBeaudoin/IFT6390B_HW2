import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m, neg_label=-1.):
        """
        y : numpy array of shape (n,) -> Values in N
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        n = y.shape[0]
        I = neg_label * np.ones((n, m))
        I[range(n), y] = 1.
        return I

    def compute_loss(self, x, y, w=None):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
            -> Values in {-1, +1}
        returns : float
        """
        scores = self.score(x, w=w)
        elementwise_loss = np.maximum(0, 2 - scores * y) ** 2
        return np.mean(elementwise_loss.sum(axis=1))

    def score(self, x, w=None):
        if w is None:
            w = self.w
        return x.dot(w)

    def compute_gradient(self, x, y, validate=False):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
            -> Values in {-1, +1}
        returns : numpy array of shape (num_features, num_classes)
        """
        n, m = y.shape
        d = x.shape[1]
        xi = self.score(x) * y
        #if self.verbose: print(f"xi:\n{xi[:5]}")
        row_effects = -2. * (xi <= 2) * y * (2 - xi)
        #if self.verbose: print(f"row_effects:\n{row_effects[:5]}")
        assert (dims := row_effects.shape) == (n, m), f"Shape: {dims}"
        data_grad = 1. / n * x.T.dot(row_effects)
        assert (dims := data_grad.shape) == (d, m), f"Shape: {dims}"
        reg_term = self.C * self.w
        if validate:
            approx = self.approximate_gradient(x, y)
            if not np.allclose(data_grad, approx, atol=1e-3):
                max_error = np.absolute(data_grad - approx).max().max()
                print("\n\n".join([
                    "Error in gradient computation!",
                    f"Analytical:\n{data_grad}",
                    f"Approximated:\n{approx}",
                    f"Max error: {max_error}"
                ]))
                raise ValueError("Mistake with gradient!")
        return data_grad + reg_term
    
    def approximate_gradient(self, x, y, eps=1e-8):
        d, m = self.w.shape
        grad = np.zeros((d, m))
        def temp_w(k, j, direction):
            w = self.w.copy()
            w[k, j] += direction * eps
            return w
        for k in range(d):
            for j in range(m):
                l1 = self.compute_loss(x, y, w=temp_w(k, j,  1.))
                l0 = self.compute_loss(x, y, w=temp_w(k, j, -1.))
                grad[k, j] = (l1 - l0) / (2 * eps)
        return grad

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        scores = self.score(x)
        y_inferred = np.argmax(scores, axis=1)
        return self.make_one_versus_all_labels(y_inferred, self.m)

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        return np.mean(np.all(y_inferred == y, axis=1))

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of testing examples, num_features)
        y_test : numpy array of shape (number of testing examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                try:
                    grad = self.compute_gradient(x, y, validate=(iteration==0))
                except RuntimeWarning as e:
                    print("\n".join([
                        f"RuntimeWarning at iteration {iteration}",
                        f"Current weight matrix:\n{self.w}"
                    ]))
                    raise e
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(" | ".join([
                    f"Iteration {iteration}",
                    f"Train loss {train_loss:.04f}",
                    f"Train acc {train_accuracy:.04f}",
                    f"Test loss {test_loss:.04f}",
                    f"Test acc {test_accuracy:.04f}"
                ]))

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        for value in ["train_losses", "train_accs", "test_losses", "test_accs"]:
            self.__dict__[value] = eval(value)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
# Data should be downloaded from the below url, and the
# unzipped folder should be placed in the same directory
# as your solution file:.
def load_data():
    # Load the data files
    print("Loading data...")
    data_path = "Star_classification/"
    dataset = pd.read_csv(data_path + "star_classification.csv")
    y = dataset['class']
    x = dataset.drop(['class','rerun_ID'], axis=1)
    
    #we replace the dataset class with a number (the class are : 'GALAXY' 'QSO' 'STAR')
    y = y.replace('GALAXY', 0)
    y = y.replace('QSO', 1)
    y = y.replace('STAR', 2)

    #split dataset in train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=40)

    #convert sets to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test=np.array(y_test)

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    print("Fitting the model...")
    svm = SVM(eta=0.0001, C=2, niter=200, batch_size=100, verbose=False)
    train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 3) # one-versus-all labels
    # svm.w = np.zeros([x_train.shape[1], 3])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
