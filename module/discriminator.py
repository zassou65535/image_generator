#encoding:utf-8
from importer import *
from generator import *
from self_attention import *

class Discriminator(nn.Module):
	def __init__(self,z_dim=20,image_size=64):
		super(Discriminator,self).__init__()

		self.layer1 = nn.Sequential(
				nn.utils.spectral_norm(nn.Conv2d(3,image_size,kernel_size=4,stride=2,padding=1)),
				nn.LeakyReLU(0.1,inplace=True))
		self.layer2 = nn.Sequential(
				nn.utils.spectral_norm(nn.Conv2d(image_size,image_size*2,kernel_size=4,stride=2,padding=1)),
				nn.LeakyReLU(0.1,inplace=True))
		self.layer3 = nn.Sequential(
				nn.utils.spectral_norm(nn.Conv2d(image_size*2,image_size*4,kernel_size=4,stride=2,padding=1)),
				nn.LeakyReLU(0.1,inplace=True))
		self.self_attention1 = Self_Attention(in_dim=image_size*4)
		self.layer4 = nn.Sequential(
				nn.utils.spectral_norm(nn.Conv2d(image_size*4,image_size*8,kernel_size=4,stride=2,padding=1)),
				nn.LeakyReLU(0.1,inplace=True))
		self.self_attention2 = Self_Attention(in_dim=image_size*8)
		self.last = nn.Sequential(
				nn.Conv2d(image_size*8,1,kernel_size=4,stride=1))

	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out,attention_map1 = self.self_attention1(out)
		out = self.layer4(out)
		out,attention_map2 = self.self_attention2(out)
		out = self.last(out)
		return out,attention_map1,attention_map2

# #動作確認
# G = Generator(z_dim=20,image_size=64)
# D = Discriminator(z_dim=20,image_size=64)

# input_z = torch.randn(1,20)
# input_z = input_z.view(input_z.size(0),input_z.size(1),1,1)
# fakeimg = G(input_z)

# d_out = D(fakeimg)

# print(nn.Sigmoid()(d_out))

