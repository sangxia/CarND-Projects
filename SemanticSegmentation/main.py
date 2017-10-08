import os.path
import numpy as np
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    ret = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    return [graph.get_tensor_by_name(tensor_name) for tensor_name in \
            [vgg_input_tensor_name, \
            vgg_keep_prob_tensor_name, \
            vgg_layer3_out_tensor_name, \
            vgg_layer4_out_tensor_name, \
            vgg_layer7_out_tensor_name]]
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """


    act_func = None
    # act_func = tf.nn.relu

    deconv1 = tf.layers.conv2d_transpose(vgg_layer7_out, 2048, \
            4, strides=(2, 2), padding='same', \
            name='seg_deconv7', activation = act_func)
    layer4_decode = tf.layers.conv2d(vgg_layer4_out, 2048, \
            1, strides=(1,1), \
            name='seg_1x1_4', activation=act_func)
    skip1 = tf.add(deconv1, layer4_decode)

    deconv2 = tf.layers.conv2d_transpose(skip1, 512, \
            4, strides=(2,2), padding='same', \
            name='seg_deconv4', activation=act_func)
    layer3_decode = tf.layers.conv2d(vgg_layer3_out, 512, \
            1, strides=(1,1), \
            name='seg_1x1_3', activation=act_func)
    skip2 = tf.add(deconv2, layer3_decode)

    seg_layer = tf.layers.conv2d_transpose(skip2, num_classes, \
            16, strides=(8,8), padding='same', \
            name='seg_deconv3', activation=act_func)
   
    return seg_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(\
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainable = None
    # only train the added variables, experiments show that this trains much faster with little loss
    # in accuracy
    # trainable = [t for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if t.name.startswith('seg_')]
    # if len(trainable) == 0:
    #     # just to get around incompatibility with unit test
    #     trainable = None
    return logits, opt.minimize(cross_entropy_loss, var_list=trainable), cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, \
        cross_entropy_loss, input_image, \
        correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())
    lr = 1e-5
    for i in range(epochs):
        epoch_loss = 0.
        epoch_total = 0
        print('epoch', i+1, 'lr', lr)
        for input_images, gt_images in get_batches_fn(batch_size):
            # use label smoothing when training
            sess.run([train_op], {\
                    input_image: input_images, \
                    correct_label: gt_images, \
                    #np.clip(gt_images, 0.05, 0.95), \
                    keep_prob: 0.8, \
                    learning_rate: lr})
            loss = sess.run([cross_entropy_loss], {\
                    input_image: input_images, \
                    keep_prob: 1., \
                    correct_label: gt_images})
            epoch_loss += loss[0]*input_images.shape[0]
            epoch_total += input_images.shape[0]
        lr = lr * 0.95
        print('{:.3f}'.format(epoch_loss / epoch_total))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
                load_vgg(sess, vgg_path)
        seg = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, loss = optimize(seg, correct_label, learning_rate, num_classes)

        train_nn(sess, 16, 10, get_batches_fn, train_op, \
                loss, input_image, correct_label, keep_prob, learning_rate)

        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, num_images=10)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
