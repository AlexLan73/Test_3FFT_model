import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
OUT='/sessions/eager-exciting-hypatia/mnt/Patent/'
plt.rcParams['font.family']='DejaVu Sans'

def new(W,H):
    f,a=plt.subplots(figsize=(W/100,H/100),dpi=140); a.set_xlim(0,W); a.set_ylim(H,0); a.axis('off'); return f,a
def tbox(a,txt,x,y,w,h,fs=12,fc='white',bold=False,lw=1.6):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=6',
                fc=fc,ec='#222222',lw=lw))
    a.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,
           fontweight='bold' if bold else 'normal',linespacing=1.35)
def arr(a,pts,lw=1.4,ls='-'):
    for p,q in zip(pts,pts[1:]): a.plot([p[0],q[0]],[p[1],q[1]],color='#222',lw=lw,ls=ls)
    p,q=pts[-2],pts[-1]; d=np.array(q,float)-np.array(p,float); d/=np.linalg.norm(d)
    a.annotate('',xy=q,xytext=tuple(np.array(q,float)-d),
               arrowprops=dict(arrowstyle='-|>',color='#222',lw=lw,mutation_scale=14,ls=ls))
IT=dict(fontsize=10.5,style='italic',color='#333')
G='#eef2f7'; Y='#fdf3e0'; GR='#e9f3ea'; R='#fbeceb'

# ============ РИС. A — когнитивный такт (глава 0) ============
W,H=1180,600; f,a=new(W,H)
tbox(a,'Зондирование\nЛЧМ  /  АМ  /  ФМ',60,60,220,60,12,G)
tbox(a,'Сменный интерфейс\n(заполнение объёма)',350,60,270,60,12,G)
tbox(a,'Общий объём\nn×n×L',690,60,180,60,12,G,bold=True)
tbox(a,'ТОКЕНИЗАТОР\nпики + страддл-инвариантные признаки',340,190,520,66,12,Y,bold=True)
tbox(a,'ИИ-гейт\n(куда смотреть)',110,330,220,60,12,G)
tbox(a,'ПРИЧИННОСТНЫЙ АРБИТР\nτ ≥ 0  /  свежесть кода',430,330,380,60,12,GR,bold=True)
tbox(a,'LSTM-добор\n(трудные)',880,330,200,60,12,G)
tbox(a,'Многолучевой агильный опрос FM-m\nкод 2ⁿ, меняется каждый такт',340,470,470,60,12,R,bold=True)
tbox(a,'Целеуказание',880,470,200,60,12,GR,bold=True)
arr(a,[(280,90),(350,90)]); arr(a,[(620,90),(690,90)])
arr(a,[(780,120),(780,160),(600,160),(600,190)])
arr(a,[(420,256),(420,295),(220,295),(220,330)])
arr(a,[(330,360),(430,360)])
arr(a,[(700,256),(700,295),(620,295),(620,330)])
arr(a,[(810,360),(880,360)])
arr(a,[(980,390),(980,430),(760,430),(760,470)])
arr(a,[(620,390),(620,470)])
arr(a,[(810,500),(880,500)])
arr(a,[(1080,500),(1140,500),(1140,20),(170,20),(170,60)])
a.text(640,12,'следующий такт: новый код, новые площади',ha='center',**IT)
a.text(890,170,'плоскость = частный случай L = 1',**IT)
a.text(60,448,'дешёвый инвариант\nгейтит дорогой зонд',fontsize=10.5,style='italic',color='#333',linespacing=1.3)
plt.savefig(OUT+'art_takt.png',dpi=140,bbox_inches='tight',facecolor='w'); plt.close()

# ============ РИС. B — объёмный примитив, смена L внутри такта (глава 4-бис) ============
W,H=1100,610; f,a=new(W,H)
a.text(40,30,'Одни и те же принятые данные такта в памяти GPU  (ось дальности R)',fontsize=12.5,fontweight='bold')
y0=76
a.text(40,66,'шаг 1 — грубый проход крупным блоком: «есть/нет и примерно где»',**IT)
for i2 in range(4):
    x=40+i2*220
    a.add_patch(Rectangle((x,y0),220,50,fc=Y if i2 in (0,2) else G,ec='#222',lw=2.2 if i2 in (0,2) else 1.5))
    a.text(x+110,y0+25,'16×16×256',ha='center',va='center',fontsize=12.5)
a.text(210,y0+66,'яркий',fontsize=11,style='italic',color='#a06000')
a.text(650,y0+66,'яркий',fontsize=11,style='italic',color='#a06000')
tbox(a,'отбор top-N наиболее ярких выбросов',300,200,480,50,12.5,Y)
arr(a,[(150,126),(150,225),(300,225)])
arr(a,[(590,126),(590,166),(880,166),(880,225),(780,225)])
y2=330
a.text(40,320,'шаг 2 — тонкий добор ТЕХ ЖЕ участков',**IT)
a.text(600,320,'мелким блоком: разрешение по дальности',**IT)
a.add_patch(Rectangle((40,y2),880,50,fc='white',ec='#222',lw=1.2))
for x0 in (46,486):
    for i2 in range(16):
        a.add_patch(Rectangle((x0+i2*13.5,y2),13.5,50,fc=Y,ec='#222',lw=1.0))
    a.text(x0+108,y2+72,'16 блоков 16×16×16',ha='center',fontsize=11.5)
arr(a,[(540,250),(540,330)])
tbox(a,'ТОТ ЖЕ токен  →  ТОТ ЖЕ арбитр',330,480,420,52,12.5,GR,bold=True)
arr(a,[(540,380),(540,480)])
a.text(790,430,'плоскость ЛЧМ = частный\nслучай при L = 1',fontsize=11,style='italic',color='#333',linespacing=1.3)
plt.savefig(OUT+'art_objem.png',dpi=140,bbox_inches='tight',facecolor='w'); plt.close()

# ============ РИС. C — причинностный арбитр (глава 5) ============
W,H=1080,540; f,a=new(W,H)
AX=250
a.annotate('',xy=(990,AX),xytext=(80,AX),arrowprops=dict(arrowstyle='-|>',color='#222',lw=1.8,mutation_scale=18))
a.text(1002,257,'R',fontsize=15)
def bar(x,h,fc,ec,lab,labc):
    a.add_patch(Rectangle((x-11,AX-h),22,h,fc=fc,ec=ec,lw=1.8))
    a.text(x,AX-h-14,lab,ha='center',fontsize=12,color=labc,fontweight='bold')
bar(230,150,'#2e7d32','#1b4d1e','ИСТИННАЯ','#1b4d1e')
for i2,x in enumerate([400,520,650,800]):
    bar(x,90-i2*9,'#f4c9c4','#a33a2e','ложная','#a33a2e')
a.annotate('',xy=(800,66),xytext=(230,66),arrowprops=dict(arrowstyle='<|-|>',color='#555',lw=1.2,mutation_scale=13))
a.text(515,54,'задержка ретранслятора  →  сдвиг  c·τ / 2',ha='center',fontsize=12,color='#333')
a.text(230,282,'передний край',ha='center',fontsize=12,fontweight='bold',color='#1b4d1e')
a.text(560,330,'ретранслятор физически может ТОЛЬКО задержать  (τ ≥ 0)',ha='center',fontsize=12.5,color='#a33a2e')
a.text(560,358,'⇒ все ложные отметки ВСЕГДА дальше истинной',ha='center',fontsize=12.5,color='#a33a2e')
tbox(a,'Решает не мощность и не нейросеть, а ФИЗИКА:\nближний член причинностной группы = истинная цель',
     140,420,800,70,12.5,GR,bold=True)
plt.savefig(OUT+'art_arbitr.png',dpi=140,bbox_inches='tight',facecolor='w'); plt.close()

# ============ РИС. D — коррелятор FM-m (глава 6) ============
W,H=1360,470; f,a=new(W,H)
tbox(a,'принятые лучи\n(S лучей)',50,60,200,58,11.5,G)
tbox(a,'FFT',300,60,90,58,12,G)
tbox(a,'агильный код 2ⁿ\n(2⁸…2²⁰, свежий)',50,230,220,58,11.5,Y)
tbox(a,'K циклических\nсдвигов',320,230,170,58,11.5,Y)
tbox(a,'FFT',540,230,90,58,12,Y)
tbox(a,'сопряжённое\nперемножение',700,140,190,66,11.5,G,bold=True)
tbox(a,'IFFT',950,145,90,58,12,G)
tbox(a,'токенизация отклика\n3–5 максимумов',1090,140,225,66,11.5,GR,bold=True)
arr(a,[(250,89),(300,89)])
arr(a,[(390,89),(795,89),(795,140)])
arr(a,[(270,259),(320,259)]); arr(a,[(490,259),(540,259)])
arr(a,[(630,259),(795,259),(795,206)])
arr(a,[(890,174),(950,174)]); arr(a,[(1040,174),(1090,174)])
a.text(815,80,'вход',**IT); a.text(815,272,'опора',**IT)
a.text(50,350,'corr = IFFT( conj(FFT(ref)) · FFT(inp) )   —   S лучей × K гипотез задержки за один проход GPU',fontsize=12.5,color='#333')
a.text(50,390,'враг не ответит на код, которого ещё не слышал  →  «свежесть кода» как второй арбитр',fontsize=12,style='italic',color='#a33a2e')
plt.savefig(OUT+'art_corr.png',dpi=140,bbox_inches='tight',facecolor='w'); plt.close()
print('4 статейных рисунка готовы')
