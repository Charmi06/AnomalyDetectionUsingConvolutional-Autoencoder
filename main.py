import argparse
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# from matplotlib.colors import hsv_to_rgb
from model import ConvolutionalAutoencoder
from PIL import Image
from dataset import load_cifar10, inverse_multiple_labeled_images

autoencoder = ConvolutionalAutoencoder(num_neuron=256, kernal1=32, kernal2=16, shape=(32, 32, 3))
optimizer = tf.keras.optimizers.SGD(learning_rate=9.0)

parser = argparse.ArgumentParser(description='Tensorflow implementation of autoencoder for anomaly detection')
parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of training epoch')
parser.add_argument('--learning_rate', '-l', type=float, default=9.0, help='Learning rate')
parser.add_argument('--train_batch', '-trb', type=int, default=128, help='Train batch amount')
parser.add_argument('--test_batch', '-tb', type=int, default=10000, help='Test batch amount')
parser.add_argument('--num_neuron', '-nn', type=int, default=256,
                    help='Number of neurons in fully connected layer for produce codes')
parser.add_argument('--step_down', '-sd', type=int, default=40, help='Step down epoch')
parser.add_argument('--anomaly_label', '-al', type=list, default=[0, 1, 2, 3, 4, 5, 6, 8, 9], help='Anomaly label')
parser.add_argument('--mode', '-m', type=str, default='RGB', help='Load cifar10 in selected format (RGB, HSV or TUV)')
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    autoencoder = ConvolutionalAutoencoder(num_neuron=args.num_neuron,
        kernal1=32, kernal2=16, shape=(32, 32, 3))
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
    train_data, test_data = load_cifar10(
        num_batch_train=args.train_batch, num_batch_test=args.test_batch, mode=args.mode)
    
    train_loss_list = []
    t1 = time.time()
    for i in range(args.epoch):
        if i != 0 and i % args.step_down == 0:
            args.learning_rate/=2
        accumulate_train_loss = []
        for train_img, train_label in train_data.as_numpy_iterator():
            with tf.GradientTape() as tape:
                logits = autoencoder.call(train_img)
                inv_train_img = inverse_multiple_labeled_images(
                    train_img, train_label, args.anomaly_label)
                loss = tf.reduce_mean(tf.losses.mean_squared_error(
                    inv_train_img,logits))
                grads = tape.gradient(loss, autoencoder.variables)
                optimizer.apply_gradients( zip(grads, autoencoder.variables))
                    #global_step=tf.Variable(0, trainable=False, dtype=tf.int64))
                accumulate_train_loss.append(loss)
        train_loss_list.append(tf.reduce_mean(accumulate_train_loss).numpy())

        print('Epoch: {}'.format(i + 1))
        print('Training MSE Loss: {}'.format(
            tf.reduce_mean(accumulate_train_loss).numpy()))
        print('Learning rate: {}'.format(args.learning_rate))
        print('Timer: {}'.format(
            time.strftime("%H:%M:%S", time.gmtime(round(time.time() - t1, 2)))))
            # plot train_loss_list after training is complete

        accumulate_test_loss = []
        if i == args.epoch - 1:
            for test_img, test_label in test_data.as_numpy_iterator():
                logits = autoencoder.call(test_img)
                inv_test_img = inverse_multiple_labeled_images(
                    test_img, test_label, args.anomaly_label)
                test_loss = tf.losses.mean_squared_error(
                    inv_test_img, logits)
                test_loss = tf.reduce_mean(test_loss, axis=[1, 2])
                loss = tf.reduce_mean(test_loss)
                accumulate_test_loss.append(loss)
                
            print('Testing MSE Loss: {}'.format(
                tf.reduce_mean(accumulate_test_loss).numpy()))

    num_img_show = 30
    for i in range(num_img_show):
        reshape_logits = logits.numpy()
        plt.subplot(5, 6, i + 1)
        plt.imshow(reshape_logits[i, :, :, :])
    plt.show()
    for i in range(num_img_show):
        plt.subplot(5, 6, i + 1)
        plt.imshow(test_img[i, :, :, :])
    plt.show()
    plt.plot(range(args.epoch), train_loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

    np.savetxt('test_label.txt', test_label.astype(int), fmt='%i')
    np.savetxt('test_loss.txt', np.array(test_loss), fmt='%f')
# save the model weights

autoencoder.save_weights("/Users/charmi/Documents/Courses/Deep Learning/Project/Test/model.h5")