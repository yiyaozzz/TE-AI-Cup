import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# train_data = mnist.train.images  # Returns np.array
# train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
# eval_data = mnist.test.images  # Returns np.array
# eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

train_data = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
eval_data = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

# The labels are already integers, so we can use them directly
train_labels = np.asarray(y_train, dtype=np.int32)
eval_labels = np.asarray(y_test, dtype=np.int32)


index = 7
plt.imshow(train_data[index].reshape(28, 28))
print("y = " + str(np.squeeze(train_labels[index])))

print("number of training examples = " + str(train_data.shape[0]))
print("number of evaluation examples = " + str(eval_data.shape[0]))
print("X_train shape: " + str(train_data.shape))
print("Y_train shape: " + str(train_labels.shape))
print("X_test shape: " + str(eval_data.shape))
print("Y_test shape: " + str(eval_labels.shape))

# def cnn_model_fn(features, labels, mode):
#     # Input Layer
#     input_height, input_width = 28, 28
#     input_channels = 1
#     input_layer = tf.reshape(
#         features["x"], [-1, input_height, input_width, input_channels])

#     # Convolutional Layer #1 and Pooling Layer #1
#     conv1_1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[
#                                     2, 2], strides=2, padding="same")

#     # Convolutional Layer #2 and Pooling Layer #2
#     conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[
#                                     2, 2], strides=2, padding="same")

#     # Convolutional Layer #3 and Pooling Layer #3
#     conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[
#                                     2, 2], strides=2, padding="same")

#     # Convolutional Layer #4 and Pooling Layer #4
#     conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     pool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=[
#                                     2, 2], strides=2, padding="same")

#     # Convolutional Layer #5 and Pooling Layer #5
#     conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[
#                                3, 3], padding="same", activation=tf.nn.relu)
#     pool5 = tf.layers.max_pooling2d(inputs=conv5_2, pool_size=[
#                                     2, 2], strides=2, padding="same")

#     # FC Layers
#     pool5_flat = tf.contrib.layers.flatten(pool5)
#     FC1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
#     FC2 = tf.layers.dense(inputs=FC1, units=4096, activation=tf.nn.relu)
#     FC3 = tf.layers.dense(inputs=FC2, units=1000, activation=tf.nn.relu)

#     """the training argument takes a boolean specifying whether or not the model is currently
#     being run in training mode; dropout will only be performed if training is true. here,
#     we check if the mode passed to our model function cnn_model_fn is train mode. """
#     dropout = tf.layers.dropout(
#         inputs=FC3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

#     # Logits Layer or the output layer. which will return the raw values for our predictions.
#     # Like FC layer, logits layer is another dense layer. We leave the activation function empty
#     # so we can apply the softmax
#     logits = tf.layers.dense(inputs=dropout, units=10)

#     # Then we make predictions based on raw output
#     predictions = {
#         # Generate predictions (for PREDICT and EVAL mode)
#         # the predicted class for each example - a vlaue from 0-9
#         "classes": tf.argmax(input=logits, axis=1),
#         # to calculate the probablities for each target class we use the softmax
#         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#     }

#     # so now our predictions are compiled in a dict object in python and using that we return an estimator object
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

#     '''Calculate Loss (for both TRAIN and EVAL modes): computes the softmax entropy loss.
#     This function both computes the softmax activation function as well as the resulting loss.'''
#     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

#     # Configure the Training Options (for TRAIN mode)
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#         train_op = optimizer.minimize(
#             loss=loss, global_step=tf.train.get_global_step())

#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

#     # Add evaluation metrics (for EVAL mode)
#     eval_metric_ops = {
#         "accuracy": tf.metrics.accuracy(labels=labels,
#                                         predictions=predictions["classes"])}
#     return tf.estimator.EstimatorSpec(mode=mode,
#                                       loss=loss,
#                                       eval_metric_ops=eval_metric_ops)
