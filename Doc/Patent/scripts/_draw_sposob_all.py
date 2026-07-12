import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
OUT='/sessions/eager-exciting-hypatia/mnt/Patent/'
BW,BH=92,44
def new(W,H):
    f,a=plt.subplots(figsize=(W/100,H/100),dpi=130); a.set_xlim(0,W); a.set_ylim(H,0); a.axis('off'); return f,a
def box(a,t,x,y,w=BW,h=BH,fs=18):
    a.add_patch(Rectangle((x,y),w,h,fill=False,ec='k',lw=1.7)); a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def line(a,pts,arrow=True):
    for p,q in zip(pts,pts[1:]): a.plot([p[0],q[0]],[p[1],q[1]],'k',lw=1.3)
    if arrow:
        p,q=pts[-2],pts[-1]; d=np.array(q,float)-np.array(p,float); d/=np.linalg.norm(d)
        a.annotate('',xy=q,xytext=tuple(np.array(q,float)-d),arrowprops=dict(arrowstyle='-|>',color='k',lw=1.3,mutation_scale=14))
def cap(a,t,W,y): a.text(W/2,y,t,ha='center',fontsize=18,fontweight='bold')
IT=dict(fontsize=11,style='italic')

# ================= ФИГ. 1 — общий такт, позиции 1–18 =================
W,H=1130,890; f,a=new(W,H)
MX=470; cx=MX+BW/2
ys={'1':30,'2':100,'3':170,'4':240,'5':310,'6':380,'7':450,'8':520,'9':590,'11':660,'15':790}
for t,y in ys.items(): box(a,t,MX,y)
for y in (30,100,170,240,310,380,450,520,590):
    if y!=660: line(a,[(cx,y+BH),(cx,y+70)])
box(a,'10',230,520); box(a,'12',650,730); box(a,'13',780,730); box(a,'14',910,730)
box(a,'16',100,130); box(a,'17',100,200); box(a,'18',100,270)

# 8 -> 10 -> 9
line(a,[(MX,542),(322,542)]); a.text(396,506,'трудный',ha='center',fontsize=10,style='italic')
line(a,[(276,564),(276,612),(MX,612)])
# 11 -> 12 -> 13 -> 14
line(a,[(562,682),(606,682),(606,752),(650,752)])
line(a,[(742,752),(780,752)]); line(a,[(872,752),(910,752)])
# 14 -> 15 ; 14 -> 9
line(a,[(956,774),(956,812),(562,812)])
line(a,[(1002,752),(1046,752),(1046,612),(562,612)])
a.text(700,604,'подтверждение кодом',**IT)
# АМ-ветвь 1 -> 16 -> 17 -> 18 -> 5
line(a,[(MX,52),(146,52),(146,130)]); a.text(240,42,'АМ-сигнал',**IT)
line(a,[(146,174),(146,200)]); line(a,[(146,244),(146,270)])
line(a,[(146,314),(146,332),(MX,332)]); a.text(200,322,'объёмный спектр',**IT)
# обратная связь
line(a,[(MX,812),(40,812),(40,14),(cx,14),(cx,30)])
a.text(28,430,'следующий такт',rotation=90,va='center',**IT)
cap(a,'Фиг. 1',W,868); plt.savefig(OUT+'Способ.png',dpi=130,bbox_inches='tight',facecolor='w'); plt.close()

# ================= ФИГ. 2 — структура токена (5, 6, 7) =================
W,H=1020,450; f,a=new(W,H)
gx,gy,gs=50,100,15
for i in range(8):
    for j in range(8): a.add_patch(Rectangle((gx+j*gs,gy+i*gs),gs,gs,fill=False,ec='k',lw=0.5))
a.add_patch(Rectangle((gx+4*gs,gy+3*gs),gs,gs,fc='k',ec='k'))
a.add_patch(Rectangle((gx+5*gs,gy+3*gs),gs,gs,fc='0.6',ec='k'))
a.text(gx+60,gy-14,'угловой срез n×n',ha='center',fontsize=12,style='italic')
box(a,'5',250,138); box(a,'6',420,138); box(a,'7',590,138)
line(a,[(gx+8*gs,160),(250,160)])
line(a,[(342,160),(420,160)]); line(a,[(512,160),(590,160)])
a.text(296,128,'до пяти пиков',ha='center',fontsize=10,style='italic')
a.text(466,128,'шесть признаков',ha='center',fontsize=10,style='italic')
for i in range(5): a.add_patch(Rectangle((780+i*7,138+i*7),110,44,fill=False,ec='k',lw=1.2))
a.text(835,128,'токены',ha='center',fontsize=11,style='italic')
line(a,[(682,160),(780,160)])
a.annotate('',xy=(890,290),xytext=(890,218),arrowprops=dict(arrowstyle='-|>',color='k',lw=1.3,mutation_scale=14))
a.text(902,258,'R',fontsize=14)
a.text(700,320,'матрица токенов по дальности',fontsize=11,style='italic')
cap(a,'Фиг. 2',W,410); plt.savefig(OUT+'Способ_фиг2.png',dpi=130,bbox_inches='tight',facecolor='w'); plt.close()

# ================= ФИГ. 3 — объёмный примитив (16, 17, 18) =================
W,H=1070,600; f,a=new(W,H)
a.text(40,28,'Принятые данные одного такта в памяти графического процессора (ось дальности R)',fontsize=12)
y0=76
a.text(40,66,'грубый проход участками увеличенной глубины',**IT)
for i in range(4):
    x=40+i*215; a.add_patch(Rectangle((x,y0),215,46,fill=False,ec='k',lw=1.7))
    a.text(x+107,y0+23,'16×16×256',ha='center',va='center',fontsize=13)
box(a,'16',940,y0+1,80,44)
box(a,'17',450,190,100,44)
a.text(500,180,'отбор наиболее ярких выбросов',ha='center',**IT)
line(a,[(147,122),(147,212),(450,212)])
line(a,[(685,122),(685,212),(550,212)])
y2=320
a.text(40,310,'тонкий добор ТЕХ ЖЕ участков',**IT)
a.text(600,310,'мелким шагом (повторное 3D-БПФ)',**IT)
a.add_patch(Rectangle((40,y2),860,46,fill=False,ec='k',lw=1.2))
for x0 in (92,630):
    for i in range(8): a.add_patch(Rectangle((x0+i*14,y2),14,46,fc='0.85',ec='k',lw=1.2))
    a.text(x0+56,y2+64,'16×16×16',ha='center',fontsize=12)
box(a,'18',940,y2+1,80,44)
line(a,[(500,234),(500,320)])
box(a,'5',454,455)
line(a,[(500,366),(500,455)])
a.text(560,430,'та же токенизация',**IT)
cap(a,'Фиг. 3',W,565); plt.savefig(OUT+'Способ_фиг3.png',dpi=130,bbox_inches='tight',facecolor='w'); plt.close()

# ================= ФИГ. 4 — причинностный арбитр (9) =================
W,H=1010,470; f,a=new(W,H)
AX=210
a.annotate('',xy=(930,AX),xytext=(90,AX),arrowprops=dict(arrowstyle='-|>',color='k',lw=1.5,mutation_scale=17))
a.text(944,216,'R',fontsize=16)
def resp(x,h,lab,fill):
    a.add_patch(Rectangle((x-9,AX-h),18,h,fc='k' if fill else 'w',ec='k',lw=1.5))
    a.text(x,AX-h-12,lab,ha='center',fontsize=13)
resp(230,120,'И',True)
for i2,x in enumerate([380,480,590,710]): resp(x,72-i2*7,'Л',False)
# размах задержки — НАД столбиками
a.annotate('',xy=(710,52),xytext=(230,52),arrowprops=dict(arrowstyle='<|-|>',color='k',lw=1.1,mutation_scale=12))
a.text(470,42,'c·τ / 2',ha='center',fontsize=13)
# подписи под осью
a.text(230,238,'истинная цель —',ha='center',**IT)
a.text(230,254,'ближний край',ha='center',**IT)
a.text(495,296,'ложные отметки ретранслятора — всегда ДАЛЬШЕ (τ ≥ 0)',ha='center',**IT)
# нижний ряд: 7 -> 9 <- (14)
box(a,'7',120,340); box(a,'9',460,340)
line(a,[(212,362),(460,362)]); a.text(250,352,'матрица токенов',**IT)
line(a,[(800,AX),(800,362),(552,362)]); a.text(600,352,'согласование с кодом (14)',**IT)
cap(a,'Фиг. 4',W,440); plt.savefig(OUT+'Способ_фиг4.png',dpi=130,bbox_inches='tight',facecolor='w'); plt.close()
print('ok')
