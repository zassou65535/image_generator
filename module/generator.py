#encoding:utf-8

from .importer import *
from .self_attention import *

class Generator(nn.Module):
	def __init__(self,z_dim=20,image_size=64):
		super(Generator,self).__init__()

		self.layer1 = nn.Sequential(
				nn.utils.spectral_norm(nn.ConvTranspose2d(z_dim,image_size*8,kernel_size=4,stride=1)),
				nn.BatchNorm2d(image_size*8),#引数はnum_features 正規化を行って平均と分散を揃える
				nn.ReLU(inplace=True))
		self.layer2 = nn.Sequential(
				nn.utils.spectral_norm(nn.ConvTranspose2d(image_size*8,image_size*4,kernel_size=4,stride=2,padding=1)),
				nn.BatchNorm2d(image_size*4),
				nn.ReLU(inplace=True))
		self.layer3 = nn.Sequential(
				nn.utils.spectral_norm(nn.ConvTranspose2d(image_size*4,image_size*2,kernel_size=4,stride=2,padding=1)),
				nn.BatchNorm2d(image_size*2),
				nn.ReLU(inplace=True))
		self.self_attention1 = Self_Attention(in_dim=image_size*2)
		self.layer4 = nn.Sequential(
				nn.utils.spectral_norm(nn.ConvTranspose2d(image_size*2,image_size,kernel_size=4,stride=2,padding=1)),
				nn.BatchNorm2d(image_size),
				nn.ReLU(inplace=True))
		self.self_attention2 = Self_Attention(in_dim=image_size)
		self.last = nn.Sequential(
				nn.ConvTranspose2d(image_size,3,kernel_size=4,stride=2,padding=1),
				nn.Tanh())

	def forward(self,z):
		out = self.layer1(z)
		out = self.layer2(out)
		out = self.layer3(out)
		out,attention_map1 = self.self_attention1(out)
		out = self.layer4(out)
		out,attention_map2 = self.self_attention2(out)
		out = self.last(out)
		return out,attention_map1,attention_map2

# #動作検証
# G = Generator(z_dim=20,image_size=64)
# input_z = torch.randn(1,20)
# input_z = input_z.view(input_z.size(0),input_z.size(1),1,1)
# img = G(input_z)
# print(img.shape)
# img_transformed = img[0].detach().numpy().transpose(1,2,0)
# print(img_transformed.shape)
# plt.imshow(img_transformed)
# plt.show()








