import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
OUT='/sessions/eager-exciting-hypatia/mnt/Patent/'
W,H=1020,600
f,a=plt.subplots(figsize=(W/100,H/100),dpi=130); a.set_xlim(0,W); a.set_ylim(H,0); a.axis('off')
def box(t,x,y,w=100,h=52,fs=17):
    a.add_patch(Rectangle((x,y),w,h,fill=False,ec='k',lw=1.6)); a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def line(pts,arrow=True):
    for p,q in zip(pts,pts[1:]): a.plot([p[0],q[0]],[p[1],q[1]],'k',lw=1.2)
    if arrow:
        p,q=pts[-2],pts[-1]; d=np.array(q)-np.array(p); d=d/np.linalg.norm(d)
        a.annotate('',xy=q,xytext=tuple(np.array(q)-d),arrowprops=dict(arrowstyle='-|>',color='k',lw=1.2,mutation_scale=13))

a.text(40,32,'Принятые данные одного такта в памяти графического процессора (ось дальности R)',fontsize=13)

# --- 3а.1 : грубый проход крупными участками ---
y0=70
a.text(40,62,'грубый проход крупным шагом',fontsize=12,style='italic')
for i in range(4):
    x=40+i*215
    a.add_patch(Rectangle((x,y0),215,46,fill=False,ec='k',lw=1.6))
    a.text(x+107,y0+23,'16×16×256',ha='center',va='center',fontsize=13)
a.text(925,y0+23,'3а.1',fontsize=16,va='center')

# --- 3а.2 : отбор top-N ---
box('3а.2',420,180,140,48,16)
a.text(490,168,'отбор наиболее ярких выбросов (top-N)',ha='center',fontsize=12,style='italic')
line([(147,116),(147,204),(420,204)])
line([(685,116),(685,204),(560,204)])

# --- 3а.3 : тонкий добор тех же участков мелким шагом ---
y2=300
a.text(40,292,'тонкий добор ТЕХ ЖЕ участков мелким шагом (повторное трёхмерное преобразование Фурье)',fontsize=12,style='italic')
a.add_patch(Rectangle((40,y2),860,46,fill=False,ec='k',lw=1.2))
for x0 in (92,630):
    for i in range(8):
        a.add_patch(Rectangle((x0+i*14,y2),14,46,fc='0.85',ec='k',lw=1.2))
    a.text(x0+56,y2+66,'16×16×16',ha='center',fontsize=12)
a.text(925,y2+23,'3а.3',fontsize=16,va='center')
line([(490,228),(490,300)])

# --- в токен ---
box('4',440,430,100,52)
line([(490,346),(490,430)])
a.text(560,412,'тот же структурный токен',fontsize=13,style='italic')

a.text(W/2,555,'Фиг. 3',ha='center',fontsize=17,fontweight='bold')
plt.savefig(OUT+'Способ_фиг3.png',dpi=130,bbox_inches='tight',facecolor='w'); print('ok')
