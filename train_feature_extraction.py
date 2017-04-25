import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import time
from alexnet import AlexNet

learning_rate = 0.001
num_epochs = 10
batch_size = 32

# TODO: Load traffic signs data.
file = open('train.p', 'rb')
dataset = pickle.load(file)
features = dataset['features']
sizes = dataset['sizes']
coords = dataset['coords']
labels = dataset['labels']
n_classes = len(np.unique(labels))
print("dataset size:", len(features))
print("image shapes:", features[0].shape)
print("sample of sizes :", sizes[:4])
print("possible labels:", np.unique(labels))
print("number of classes:", n_classes)

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size =0.3) 
print("size of training set:", len(X_train))
print("size of validation set:", len(X_valid))
print("size of testing set:", len(X_test))
# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y_labels = tf.placeholder(tf.int64, (None))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], n_classes)  # use this shape for the weight matrix
W_last = tf.Variable(tf.random_normal(shape, mean=0, stddev=1e-2))
b_last = tf.Variable(tf.zeros(n_classes))
logits = tf.nn.xw_plus_b(fc7, W_last, b_last)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_labels)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(loss, var_list=[W_last, b_last])

# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
# accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, y_labels), tf.float32))

# The evaluation function
def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        ls, acc = sess.run([loss, accuracy_op], feed_dict={x: X_batch, y_labels: y_batch})
        total_loss += (ls * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]


num_examples = len(X_train)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(training, feed_dict={x: X_train[offset:end], y_labels: y_train[offset:end]})

        val_loss, val_acc = eval_on_data(X_valid, y_valid, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(num_epochs):
		X_train, y_train = shuffle(X_train, y_train)
		t0 = time.time()
		for offset in range(0, num_examples, batch_size):
		    end = offset + batch_size
		    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
		    sess.run(training, feed_dict={x:X_train, y_labels:y_train })

		validation_accuracy = eval_on_data(X_valid, y_valid)
		print("EPOCH {} ...".format(i+1))
		print("Time: %.3f seconds" % (time.time() - t0))
		print("Validation Accuracy = {:.3f}".format(validation_accuracy))
		print()
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
# TODO: Train and evaluate the feature extraction model.
