import tensorflow as tf
import tensorflow_datasets as tfds
import datetime

cifar, cifar_info = tfds.load("cifar10", with_info=True)

cifar_train, cifar_test = cifar['train'], cifar['test']

# cifar_train set has 50000 samples

# Create dataset pipeline:
assert isinstance(cifar_train, tf.data.Dataset)
cifar_train = cifar_train.shuffle(buffer_size=50000)
cifar_train = cifar_train.batch(batch_size=100)


def conv_block(units, previous_block_units=None, first_block=False):

    if first_block is True:
        block_input = tf.keras.layers.Input(shape=(32, 32, 3))
        conv_1 = tf.keras.layers.Conv2D(units, 5, strides=(1, 1), input_shape=(None, 32, 32, 3), padding='same')\
            (block_input)
        conv_2 = tf.keras.layers.Conv2D(units, 5, strides=(1, 1), padding='same')(conv_1)
    else:
        block_input = tf.keras.layers.Input(shape=(16, 16, previous_block_units))
        conv_1 = tf.keras.layers.Conv2D(units, 3, strides=(1, 1), padding='same')(block_input)
        conv_2 = tf.keras.layers.Conv2D(units, 3, strides=(1, 1), padding='same')(conv_1)

    batch_norm = tf.keras.layers.BatchNormalization()(conv_2)
    activation = tf.keras.layers.Activation("relu")(batch_norm)
    block_output = tf.keras.layers.MaxPooling2D()(activation)

    return tf.keras.Model(inputs=block_input, outputs=block_output)


class ConvCifar10(tf.keras.Model):
    """Remember Cifar10 has images of shape (32, 32, 3)."""

    def __init__(self, number_of_classes=10):
        super().__init__()

        # todo: try performance with different number of layers:
        # TODO: add RESNET
        self.conv_block_1 = conv_block(32, first_block=True)
        self.conv_block_2 = conv_block(64, previous_block_units=32)
        self.conv_block_3 = conv_block(128, previous_block_units=64)

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(number_of_classes)  # We don't add softmax activation since it is already
        # computed in the CategoricalCrossentropy loss_object later on.

    def __call__(self, inputs):

        x = self.conv_block_1(inputs)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        output = self.dense_2(x)

        return output


# Instantiate the model:
convcifar_model = ConvCifar10()

# Tensorboard  summaries:
current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
summary_dir = './logs/cifar10conv/train_2convx3deep_increasing_units_0.01-lr_' + current_time
summary_writer = tf.summary.create_file_writer(summary_dir)

# Establish loss, optimizer, and metrics:
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # todo: we could schedule the learning rate

loss_metric = tf.keras.metrics.Mean()
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()


# Saving the model:
checkpoint_path = "./checkpoints/cifar10conv/train"
checkpoint = tf.train.Checkpoint(model=convcifar_model, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)


def train_step(model_instance, inputs, labels):

    with tf.GradientTape() as tape:
        predictions = model_instance(inputs)
        # predictions = tf.reshape(predictions, shape=(10, 100))
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model_instance.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_instance.trainable_variables))  # (gradient, variables)

    loss_metric(loss)
    accuracy_metric(labels, predictions)


# Training loop --------------------------------------------------------------------------------------------------------

EPOCHS = 20

for epoch in range(EPOCHS):

    for batch, dataset in enumerate(cifar_train):
        inputs, labels = dataset.values()

        # A couple changes to inputs and labels:
        labels = tf.one_hot(labels, 10)
        inputs = tf.cast(inputs, tf.float32)  # Otherwise they are tf.unint8

        train_step(convcifar_model, inputs, labels)

        if (batch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, batch {batch + 1} => "
                  f"Loss {loss_metric.result():.4f}  Accuracy {accuracy_metric.result():.4f}")

        # Write summaries for tensorboard:
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss_metric.result(), step=batch)
            tf.summary.scalar("accuracy", accuracy_metric.result(), step=batch)

    print("-" * 80)
    print(f"Epoch {epoch +1}: Loss {loss_metric.result():.4f} Accuracy {accuracy_metric.result():.4f}")

    # Saving the model:
    if (epoch + 1) % 5 == 0:
        checkpoint_manager.save()
        print(f"Checkpoint saved!")





