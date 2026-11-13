import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Definir transformación para preprocesamiento de imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar el dataset CIC-IDS2017
train_data = datasets.ImageFolder('ruta/al/dataset', transform=transform)

# Definir el tamaño del batch y crear los dataloaders
batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Definir la dimensión del ruido de entrada
noise_dim = 100

# Definir el generador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 52)
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

# Definir el discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(52, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instanciar el generador y el discriminador
generator = Generator()
discriminator = Discriminator()

# Definir la función de pérdida y los optimizadores
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# Función para generar ruido aleatorio
def generate_noise(n_samples):
    return torch.randn(n_samples, noise_dim)

# Función para entrenar el discriminador
def train_discriminator(real_samples, dis_optimizer):
    # Etiquetar las muestras reales con 1
    real_labels = torch.ones(real_samples.size(0), 1)
    # Pasar las muestras reales por el discriminador
    real_output = discriminator(real_samples)
    # Calcular la pérdida del discriminador para las muestras reales
    real_loss = criterion(real_output, real_labels)
    
    # Generar ruido aleatorio y pasar por el generador para obtener muestras sintéticas
    fake_noise = generate_noise(real_samples.size(0))
    fake_samples = generator(fake_noise)
    # Etiquetar las muestras sintéticas con 0
    fake_labels = torch.zeros(fake_samples.size(0), 1)
    # Pasar las muestras sintéticas por el discriminador
    fake_output = discriminator(fake_samples)
    # Calcular la pérdida del discriminador para las muestras sintéticas
    fake_loss = criterion(fake_output, fake_labels)

    # Sumar las pérdidas y propagar hacia atrás
    dis_loss = real_loss + fake_loss
    dis_optimizer.zero_grad()
    dis_loss.backward()
    dis_optimizer.step()

    return dis_loss.item()

def train_generator(gen_optimizer):
    # Generar ruido aleatorio y pasar por el generador para obtener muestras sintéticas
    fake_noise = generate_noise(batch_size)
    fake_samples = generator(fake_noise)
    # Etiquetar las muestras sintéticas con 1 (tratando de engañar al discriminador)
    fake_labels = torch.ones(batch_size, 1)
    # Pasar las muestras sintéticas por el discriminador
    fake_output = discriminator(fake_samples)
    # Calcular la pérdida del generador
    gen_loss = criterion(fake_output, fake_labels)
    # Propagar hacia atrás y actualizar los pesos del generador
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    return gen_loss.item()

def train_gan(generator, discriminator, gen_optimizer, dis_optimizer, n_epochs):
    for epoch in range(n_epochs):
        dis_loss = 0
        gen_loss = 0
        for real_samples, _ in train_loader:
            # Entrenar el discriminador
            dis_loss += train_discriminator(real_samples, dis_optimizer)
        # Congelar el discriminador y entrenar el generador
        for p in discriminator.parameters():
            p.requires_grad = False
            
        gen_loss += train_generator(gen_optimizer)
        
        # Descongelar el discriminador
        for p in discriminator.parameters():
            p.requires_grad = True
            
        # Imprimir el progreso del entrenamiento en cada epoch
        print('Epoch [{}/{}], Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(epoch+1, n_epochs, dis_loss/len(train_loader), gen_loss/len(train_loader)))


n_epochs = 50
train_gan(generator, discriminator, gen_optimizer, dis_optimizer, n_epochs)