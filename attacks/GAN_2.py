import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load the CIC-IDS2017 dataset
data = pd.read_csv('CIC-IDS2017.csv')

# Preprocessing the data
data = data.dropna()
data = data.drop_duplicates()
data[' Label'] = data[' Label'].apply(lambda x: 1 if x == 'BENIGN' else 0)
X = data.drop([' Label'], axis=1)
y = data[' Label']
X = (X - X.mean()) / X.std()

# Convert the data into numpy arrays
X = X.to_numpy()
y = y.to_numpy()

# Define the generator network
generator = Sequential()
generator.add(Dense(256, input_dim=100))
generator.add(LeakyReLU(alpha=0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(alpha=0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(alpha=0.2))
generator.add(Dense(X.shape[1], activation='tanh'))
generator.add(Reshape((X.shape[1],)))

# Define the discriminator network
discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=X.shape[1]))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Define the GAN by combining the generator and discriminator
discriminator.trainable = False
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Define the training loop
epochs = 100
batch_size = 32
d_loss = []
g_loss = []
for epoch in range(epochs):
    # Select a random batch of data
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_data = X[idx]

    # Generate a batch of fake data
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_data = generator.predict(noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
    d_loss_epoch = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss_epoch = gan.train_on_batch(noise, np.ones((batch_size, 1)))
  
    d_loss.append(d_loss_epoch)
    g_loss.append(g_loss_epoch)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Discriminator loss: {d_loss_epoch}, Generator loss: {g_loss_epoch}")
        

plt.plot(d_loss, label='Discriminator Loss')
plt.plot(g_loss, label='Generator Loss')
plt.legend()
plt.show()