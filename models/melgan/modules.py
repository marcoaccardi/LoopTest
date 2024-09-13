# modules.py

from models.melgan.weights import weights_init, WNConv1d, WNConvTranspose1d
from models.melgan.audio2mel import Audio2Mel
from models.melgan.resnet import ResnetBlock
from models.melgan.generator import Generator_melgan
from models.melgan.discriminator import NLayerDiscriminator, Discriminator
