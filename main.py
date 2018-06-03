# Semantic Segmentation
#####################################################################################


import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
#import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

 
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_layer, keep_prob, layer_3, layer_4, layer_7



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    # 1x1 convolution of vgg layer 7
    layer_7_conv = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.001),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # upsample vgg layer 7
    layer_7_upsample = tf.layers.conv2d_transpose(layer_7_conv, num_classes, 4, 
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.001), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))


    # 1x1 convolution of vgg layer 4
    layer_4_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.001), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection (element-wise addition)
    layer_4_skip = tf.add(layer_7_upsample, layer_4_conv)


    # upsample
    layer_4_upsample = tf.layers.conv2d_transpose(layer_4_skip, num_classes, 4,  
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.001), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # 1x1 convolution of vgg layer 3
    layer_3_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.001), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # skip connection (element-wise addition)
    layer_3_skip = tf.add(layer_4_upsample, layer_3_conv)
    
    # upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer_3_skip, num_classes, 16,  
                                               strides= (8, 8), 
                                               padding= 'same', 
                                               kernel_initializer= tf.random_normal_initializer(stddev=0.001), 
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    return nn_last_layer


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='my_logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # define loss function and training op
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cross_entropy_loss + reg_losses

    #make a prediction
    prediction = tf.argmax(tf.nn.softmax(logits), axis=1, name='prediction_fn')

    train_op = optimizer.minimize(loss=loss)

    return logits, train_op, loss, prediction


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver, prediction, restart_training=True):

    image_shape = (160, 288)
    data_dir = './data'
    runs_dir = './runs'
    
    if(restart_training):
        sess.run(tf.global_variables_initializer())

    print("Training...")
    print()

    for epoch in range(epochs):

        print("EPOCH: {}".format(epoch+1))
        
        for image, label in get_batches_fn(batch_size):

            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, 
                                                                            keep_prob: 0.8, learning_rate: 0.0001})

            print("Loss: = {:.6f}".format(loss))
        print()

        # Save the model every 10 epochs and also save some images
        if(epoch % 10 == 0 or epoch == epochs):
            model_path = "./tmp/epoch_{}/model.ckpt".format(epoch)
            save_path = saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)

            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, prediction, keep_prob, input_image)





def run():
    num_classes = 3
    image_shape = (160, 288)
    data_dir = './data'
    runs_dir = './runs'

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        
        # get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'Train'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        epochs = 1
        batch_size = 8

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss, prediction = optimize(nn_last_layer, correct_label, learning_rate, num_classes)


        saver = tf.train.Saver()

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, saver, prediction)
        
        
        #model_path = "./tmp/model.ckpt"
        #save_path = saver.save(sess, model_path)
        #print("Model saved in file: %s" % save_path)
        

        # TODO: Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, prediction, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
