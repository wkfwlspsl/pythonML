import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# //--------------------------------------
# plt.title("line plot")
# plt.plot([10,20,30,40],[1,4,9,16],'b2-')
# plt.savefig('ex_lineplot.png', format='png', dpi=300) # 파일로 저장
# plt.show()

# font_location='C:\Windows\Fonts\malgun.ttf'
# font_name=font_manager.FontProperties(fname=font_location).get_name()
# font={'family':font_name,'color':'black','size':20}
# plt.title('라인 플롯(line plot)', fontdict=font)
# plt.plot([10,20,30,40],[1,4,9,16],c="b",lw=5,ls="--",marker="o",ms=15,mec="g",mew=5,mfc='r')
# plt.xlim(0,50)
# plt.ylim(0,20)
# plt.show()
# --------------------------------------//

# //--------------------------------------
# x1=np.linspace(0.0,5.0,num=50)
# x2=np.linspace(0.0,2.0,num=50)
# y1=np.cos(2*np.pi*x1)*np.exp(-x1)
# y2=np.cos(2*np.pi*x2)
#
# plt.subplot(121)
# plt.plot(x1, y1,'yo-')
# plt.title('A tale of 2 subplots-1')
# plt.xlabel('tune (s)1')
# plt.ylabel('Damped oscillation')
#
# plt.subplot(122)
# plt.plot(x2, y2,'r.-')
# plt.title('A tale of 2 subplots-2')
# plt.xlabel('tune (s)2')
# plt.ylabel('Undamped')
#
# plt.tight_layout()
# plt.show()
# --------------------------------------//

# //--------------------------------------
# font_location='C:\Windows\Fonts\malgun.ttf'
# font_name=font_manager.FontProperties(fname=font_location).get_name()
# font={'family':font_name,'color':'black','size':20}
# mpl.rc('font', family=font_name)
#
# fig, ax0 = plt.subplots() # 복수 개의 subplot 생성
# ax1=ax0.twinx() # twinx():x축을 공유하는 새로운 Axes객체 생성
#
# ax0.set_title("2개의 y축 한 figure에서 사용하기")
# ax0.plot([10,5,2,9,7], 'r-', label='y0')
# ax0.set_ylabel('y0')
# ax0.set_xlabel('공유되는 x축')
# ax0.legend(loc=2)
# ax0.grid(False)
#
# # ax1.set_title("2개의 y축 한 figure에서 사용하기")
# ax1.plot([100,200,220,180,120],'g:',label='y1')
# ax1.set_ylabel('y1')
# ax1.legend(loc=1)
# ax1.grid(False)
# plt.show()
# --------------------------------------//

# //--------------------------------------
font_location='C:\Windows\Fonts\malgun.ttf'
font_name=font_manager.FontProperties(fname=font_location).get_name()
font={'family':font_name,'color':'red','size':15}
font2={'family':font_name,'color':'black','size':11}

np.random.seed(0)
X=np.random.normal(0,1,100)
Y=np.random.normal(0,1,100)
plt.title('Scatter Plot(산점도)', fontdict=font)
plt.scatter(X,Y)
plt.xlabel('x축',fontdict=font2)
plt.ylabel('y축',fontdict=font2)
plt.show()
# --------------------------------------//






















