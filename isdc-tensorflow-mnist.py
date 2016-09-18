import numpy as numpy
import pandas as pandas
import tensorflow as tensorflow

print('Initializing constants...')
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 10000
DROPOUT = 0.5
BATCH_SIZE = 100
VALIDATION_SIZE = 2000
IMAGE_TO_DISPLAY = 10

TRAINING_FILE_PATH = './data/isdc_mnist_train.csv'
TEST_FILE_PATH = './data/isdc_mnist_test.csv'
SUBMISSION_FILE_PATH = './data/submission.csv'
SUBMISSION_FILE_HEADER = 'imageId,label'

TENSOR_SHAPE_CONVOLUTION_LAYER_1 = [5, 5, 1, 32]  # 5 patch size, grayscale, 32 features
TENSOR_SHAPE_CONVOLUTION_LAYER_2 = [5, 5, 32, 64]  # 5 patch size, 32 features input, 64 features output

CONVOLUTION_POOLING = [1, 4, 4, 1]

STRIDES = [1, 1, 1, 1]
OUTPUT_STRIDES = [1, 2, 2, 1]

ALL_IMAGES_FEATURES_CONVOLUTION_LAYER_2 = 7 * 7 * 64

print('Reading training data...')
training_data = pandas.read_csv(TRAINING_FILE_PATH)
images_data = training_data.iloc[:, 1:].values
images_data = images_data.astype(numpy.float)

images_data = numpy.multiply(images_data, 1.0 / 255.0)  # normalize

image_size = images_data.shape[1]
image_width = image_height = numpy.ceil(numpy.sqrt(image_size)).astype(numpy.uint8)  # square images

labels_flat = training_data[[0]].values.ravel()
labels_size = numpy.unique(labels_flat).shape[0]


def convert_labels_to_one_hot_vector(scalar_labels, number_of_classes):
    number_of_labels = scalar_labels.shape[0]
    index = numpy.arange(number_of_labels) * number_of_classes
    one_hot_labels = numpy.zeros((number_of_labels, number_of_classes))
    one_hot_labels.flat[index + scalar_labels.ravel()] = 1
    return one_hot_labels


labels_one_hot = convert_labels_to_one_hot_vector(labels_flat, labels_size)
labels_one_hot = labels_one_hot.astype(numpy.uint8)

images_validation = images_data[:VALIDATION_SIZE]
labels_validation = labels_one_hot[:VALIDATION_SIZE]

training_images = images_data[VALIDATION_SIZE:]
training_labels = labels_one_hot[VALIDATION_SIZE:]

images_nn_input = tensorflow.placeholder('float', shape=[None, image_size])
labels_nn_output = tensorflow.placeholder('float', shape=[None, labels_size])

print('Setting up 1st convolution layer...')
weight_convolution_layer_1 = tensorflow.Variable(
    tensorflow.truncated_normal(
        TENSOR_SHAPE_CONVOLUTION_LAYER_1,
        stddev=0.1
    )
)
bias_convolution_layer_1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[32]))

image = tensorflow.reshape(images_nn_input, [-1, image_width, image_height, 1])  # reshape to 4D tensor

print('Setting up activation result after 1st convolution layer...')
activation_result_convolution_layer_1 = tensorflow.nn.relu6(
    tensorflow.nn.conv2d(
        image,
        weight_convolution_layer_1,
        strides=STRIDES,
        padding='SAME'
    ) + bias_convolution_layer_1
)

print('Setting up pooling for tensor for input for 2nd convolution layer...')
output_tensor_convolution_layer_1 = tensorflow.nn.max_pool(
    activation_result_convolution_layer_1,
    ksize=CONVOLUTION_POOLING,
    strides=OUTPUT_STRIDES,
    padding='SAME'
)

print('Setting up 2nd convolution layer...')
weight_convolution_layer_2 = tensorflow.Variable(
    tensorflow.truncated_normal(
        TENSOR_SHAPE_CONVOLUTION_LAYER_2,
        stddev=0.1
    )
)
bias_convolution_layer_2 = tensorflow.Variable(
    tensorflow.constant(
        0.1,
        shape=[64]
    )
)

print('Setting up activation result after 2nd convolution layer...')
activation_result_convolution_layer_2 = tensorflow.nn.relu6(
    tensorflow.nn.conv2d(
        output_tensor_convolution_layer_1,
        weight_convolution_layer_2,
        strides=STRIDES,
        padding='SAME'
    ) + bias_convolution_layer_2
)

print('Setting up pooling for tensor for input for fully connected layer...')
output_tensor_convolution_layer_2 = tensorflow.nn.max_pool(
    activation_result_convolution_layer_2,
    ksize=CONVOLUTION_POOLING,
    strides=OUTPUT_STRIDES,
    padding='SAME')

print('Setting up fully connected layer...')
weight_fully_connected_layer = tensorflow.Variable(
    tensorflow.truncated_normal(
        [ALL_IMAGES_FEATURES_CONVOLUTION_LAYER_2, 1024],
        stddev=0.1
    )
)
bias_fully_connected_layer = tensorflow.Variable(
    tensorflow.constant(
        0.1,
        shape=[1024]
    )
)

print('Setting up tensor for final activation function filtering...')
output_tensor_flat_convolution_layer_2 = tensorflow.reshape(
    output_tensor_convolution_layer_2,
    [-1, ALL_IMAGES_FEATURES_CONVOLUTION_LAYER_2]
)
activation_result_fully_connected_layer = tensorflow.nn.sigmoid(
    tensorflow.matmul(
        output_tensor_flat_convolution_layer_2,
        weight_fully_connected_layer
    ) + bias_fully_connected_layer
)

print('Setting up dropout...')
probability_to_keep_in_network = tensorflow.placeholder('float')
output_tensor_dropout_fully_connected_layer = tensorflow.nn.dropout(
    activation_result_fully_connected_layer,
    probability_to_keep_in_network
)

print('Setting up output layer for softmax regression...')
weight_output_layer = tensorflow.Variable(
    tensorflow.truncated_normal(
        [1024, labels_size],
        stddev=0.1
    )
)
bias_output_layer = tensorflow.Variable(tensorflow.constant(0.1, shape=[labels_size]))
output_tensor_softmax_regression = tensorflow.nn.softmax(
    tensorflow.matmul(
        output_tensor_dropout_fully_connected_layer,
        weight_output_layer
    ) + bias_output_layer
)

print('Setting up reduced tensor...')
reduced_tensor_cross_entropy_cost_function = -tensorflow.reduce_sum(
    labels_nn_output * tensorflow.log(output_tensor_softmax_regression)
)

print('Setting up AdamOptimizer...')
optimized_training_step_output = tensorflow.train.AdamOptimizer(LEARNING_RATE).minimize(
    reduced_tensor_cross_entropy_cost_function
)

evaluation_output = tensorflow.equal(
    tensorflow.argmax(output_tensor_softmax_regression, 1),
    tensorflow.argmax(labels_nn_output, 1)
)

accuracy = tensorflow.reduce_mean(
    tensorflow.cast(evaluation_output, 'float')
)

prediction_output = tensorflow.argmax(output_tensor_softmax_regression, 1)

training_epochs_done = 0
epoch_index = 0
training_images_count = training_images.shape[0]


def stochastic_training_next_batch(batch_size):
    global training_images
    global training_labels
    global epoch_index
    global training_epochs_done

    start = epoch_index
    epoch_index += batch_size

    # when all training data have been already used, it is reorder randomly
    if epoch_index > training_images_count:
        training_epochs_done += 1
        permutations = numpy.arange(training_images_count)
        numpy.random.shuffle(permutations)
        training_images = training_images[permutations]
        training_labels = training_labels[permutations]
        # start next epoch
        start = 0
        epoch_index = batch_size
        assert batch_size <= training_images_count
    end = epoch_index
    return training_images[start:end], training_labels[start:end]


tensorflow_session = tensorflow.InteractiveSession()
tensorflow_session.run(tensorflow.initialize_all_variables())

training_accuracy_array = []
validation_accuracy_array = []

display_step = 1

print('Beginning training...')
for i in range(TRAINING_ITERATIONS):
    batch_images, batch_labels = stochastic_training_next_batch(BATCH_SIZE)
    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:
        train_accuracy = accuracy.eval(
            feed_dict={
                images_nn_input: batch_images,
                labels_nn_output: batch_labels,
                probability_to_keep_in_network: 1.0
            }
        )
        if VALIDATION_SIZE:
            validation_accuracy = accuracy.eval(
                feed_dict={
                    images_nn_input: images_validation[0:BATCH_SIZE],
                    labels_nn_output: labels_validation[0:BATCH_SIZE],
                    probability_to_keep_in_network: 1.0
                }
            )
            print('Accuracy training data / Accuracy validation data = %.2f / %.2f at training step %d' % (
                train_accuracy, validation_accuracy, i))
            validation_accuracy_array.append(validation_accuracy)
        else:
            print('Accuracy training data = %.4f at training step %d' % (train_accuracy, i))
            training_accuracy_array.append(train_accuracy)
        if i % (display_step * 10) == 0 and i:
            display_step *= 10

    tensorflow_session.run(
        optimized_training_step_output,
        feed_dict={
            images_nn_input: batch_images,
            labels_nn_output: batch_labels,
            probability_to_keep_in_network: DROPOUT
        }
    )

print('Reading test data...')
with open(TEST_FILE_PATH, 'r') as reader:
    reader.readline()
    images_testing = []
    for line in reader.readlines():
        if line is not '\n':
            line = line[1:]
            pixels = list(map(int, line.rstrip().split(',')))
            images_testing.append(pixels)
print('Loaded ' + str(len(images_testing)) + ' test images...')

images_testing = numpy.multiply(images_testing, 1.0 / 255.0)

predicted_labels = numpy.zeros(images_testing.shape[0])
for i in range(0, images_testing.shape[0] // BATCH_SIZE):
    predicted_labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = prediction_output.eval(
        feed_dict={
            images_nn_input: images_testing[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
            probability_to_keep_in_network: 1.0
        }
    )

print('Predicted Labels Number: ({0})'.format(len(predicted_labels)))

print('Predicted Label [{0}] => {1}'.format(IMAGE_TO_DISPLAY, predicted_labels[IMAGE_TO_DISPLAY]))

print('Saving results...')
numpy.savetxt(
    SUBMISSION_FILE_PATH,
    numpy.c_[range(1, len(images_testing) + 1), predicted_labels],
    delimiter=',',
    header=SUBMISSION_FILE_HEADER,
    comments='',
    fmt='%d'
)

tensorflow_session.close()
print('Results saved! TensorFlow session closed.')
