import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True)
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

ds = tfds.load('fashion_mnist', split='train')
import numpy as np

dataiterator = ds.as_numpy_iterator()
dataiterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
  batch = dataiterator.next()
  ax[idx].imshow(np.squeeze(batch['image']))
  ax[idx].title.set_text(batch['label'])
def scale_images(data):
  image = data['image']
  return image / 255
ds = tfds.load('fashion_mnist', split='train')
ds = ds.map(scale_images)
ds = ds.cache()
ds = ds.shuffle(6000)
ds = ds.batch(128)
ds = ds.prefetch(64)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout
from tensorflow.keras.layers import UpSampling2D

def build_generator():
    model = Sequential()

    # We pass 128 random values to generate our first images
    # That 128 is going to be converted into a spatial area bc we eventually 
    # want to generate an image so we give it 7x7x28
    # Why 7? Our final image has to be 28x28x1 and 7x2x2 = 28 
    model.add(Dense(7*7*128, input_dim = 128))
    model.add(LeakyReLU(0.2))
    # here we reshape the 6272 values (7x7x28) into an array of dimension 7x7x28
    # in the next layers we have to transform from 7x7x28 to 28x28x1
    model.add(Reshape((7,7,128))) 


    # Upsampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, 6, padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 6, padding='same'))
    model.add(LeakyReLU(0.2))


    # Convolutional block 1 
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 2
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))


    # Conv Layer to get from 128 channels  to one channel 
    model.add(Conv2D(1,4, padding='same', activation = 'sigmoid'))

    return model 
generator = build_generator()
generator.summary()
img = generator.predict(np.random.randn(4,128,1)) # 4 images with 128 random values which is what our generator takes

# Visualize random generated img

fig, ax = plt.subplots(ncols=4, figsize=(20,20))

for idx, img in enumerate(img):
  ax[idx].imshow(np.squeeze(img))
  ax[idx].title.set_text(idx)

def build_discriminator():
  model = Sequential()
  
  model.add(Conv2D(16, 5, input_shape= (28,28,1))) # input shape has to be the same as the output from our generator
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  # Second Conv Block 
  model.add(Conv2D(32, 5))   
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  # Third Conv Block 
  model.add(Conv2D(64, 5))   
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  # Fourth Conv Block
  model.add(Conv2D(128, 5))   
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  # Fifth Conv Block
  model.add(Conv2D(256, 5))   
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  model.add(Flatten())
  model.add(Dropout(0.4))
  model.add(Dense(1, activation='sigmoid')) # will return how confident between 0-1 is that our image is generated

  return model

discriminator = build_discriminator()
discriminator.summary(  )
img = generator.predict(np.random.randn(4,128,1))
img.shape

discriminator.predict(img) # prediction for 4 random generated images.
# Construct Training Loop
import tensorflow as tf
# optimizer for both
from tensorflow.keras.optimizers import Adam

# Loss for both
from tensorflow.keras.losses import BinaryCrossentropy
g_opt = Adam(learning_rate = 0.0001)
d_opt = Adam(learning_rate = 0.00001)

g_loss = tf.keras.losses.BinaryCrossentropy()
d_loss = tf.keras.losses.BinaryCrossentropy()
# Import the base model class to subclass
from tensorflow.keras.models import Model
class MyGAN(Model): 
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class 
        super().__init__(*args, **kwargs)
        
        # Create attributes for gen and disc
        self.generator = generator 
        self.discriminator = discriminator 
        
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs): 
        # Compile with base class
        super().compile(*args, **kwargs)
        
        # Create attributes for losses and optimizers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss 

    def train_step(self, batch):
        # Get the data 
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)
        
        # Train the discriminator
        with tf.GradientTape() as d_tape: 
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True) 
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            
            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)
            
            # Add some noise to the TRUE outputs
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            
            # Calculate loss - BINARYCROSS 
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)
            
        # Apply backpropagation - nn learn 
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables) 
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
        
        # Train the generator 
        with tf.GradientTape() as g_tape: 
            # Generate some new images
            gen_images = self.generator(tf.random.normal((128,128,1)), training=True)
                                        
            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)
                                        
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels) 
            
        # Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
        
        return {"d_loss":total_d_loss, "g_loss":total_g_loss}


# Create instance of subclassed model
mygan = MyGAN(generator, discriminator)

# Compile the model 
mygan.compile(g_opt, d_opt, g_loss, d_loss)
import os 
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
class ModelMonitor(Callback):
  def __init__(self, num_img = 3, latent_dim = 128):
    self.num_img = num_img
    self.latent_dim = latent_dim

  def on_epoch_end(self, epoch, logs=None):
    random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim, 1))
    generated_images = self.model.generator(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(self.num_img):
      img = array_to_img(generated_images[i])
      img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))

