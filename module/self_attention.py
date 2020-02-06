#encoding:utf-8
from .importer import *
from .generator import *

class Self_Attention(nn.Module):
	def __init__(self,in_dim):
		super(Self_Attention,self).__init__()

		#1×1の畳み込み層によるpointwise convolutionを用意
		self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
		#pythonでは//で小数点以下切り捨ての割り算をする（結果が整数になる）

		#Attention Map作成時の規格化のソフトマックス関数
		self.softmax = nn.Softmax(dim=-2)
		#係数は最初は0にしておく　徐々に学習しながら変化する
		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self,x):
		X = x
		#畳み込みをしてからサイズを変形 B,C',W,HからB,C',Nにする
		proj_query = self.query_conv(X).view(X.shape[0],-1,X.shape[2]*X.shape[3])#B,C',Nに
		proj_query = proj_query.permute(0,2,1)#転置
		proj_key = self.key_conv(X).view(X.shape[0],-1,X.shape[2]*X.shape[3])#B,C',N
		#掛け算
		S = torch.bmm(proj_query,proj_key)#bmmはバッチごとの掛け算が可能
		#規格化
		attention_map_T = self.softmax(S)#行i方向の和を1にするsoftmax関数
		attention_map = attention_map_T.permute(0,2,1)#転置を取る
		#self-attention mapを計算
		proj_value = self.value_conv(X).view(X.shape[0],-1,X.shape[2]*X.shape[3])#サイズB,C,N
		o = torch.bmm(proj_value,attention_map.permute(0,2,1))
		#self-attention mapであるoのテンソルサイズをXに揃え出力
		o = o.view(X.shape[0],X.shape[1],X.shape[2],X.shape[3])
		out = x+self.gamma*o
		return out,attention_map



