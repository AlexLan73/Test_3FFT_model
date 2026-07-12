import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
OUT='/sessions/eager-exciting-hypatia/mnt/Patent/'

def newfig(W,H):
    f,a=plt.subplots(figsize=(W/100,H/100),dpi=130); a.set_xlim(0,W); a.set_ylim(H,0); a.axis('off'); return f,a
def box(a,t,x,y,w=100,h=52,fs=17):
    a.add_patch(Rectangle((x,y),w,h,fill=False,ec='k',lw=1.6))
    a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def line(a,pts,arrow=True):
    for p,q in zip(pts,pts[1:]): a.plot([p[0],q[0]],[p[1],q[1]],'k',lw=1.2)
    if arrow:
        p,q=pts[-2],pts[-1]; d=np.array(q)-np.array(p); d=d/np.linalg.norm(d)
        a.annotate('',xy=q,xytext=tuple(np.array(q)-d),arrowprops=dict(arrowstyle='-|>',color='k',lw=1.2,mutation_scale=13))
def cap(a,t,W,y): a.text(W/2,y,t,ha='center',fontsize=17,fontweight='bold')

# ================= ФИГ. 2 — структура токена (детализация операции 4) =================
W,H=980,470; f,a=newfig(W,H)
# угловой срез n×n
gx,gy,gs=40,90,14
for i in range(8):
    for j in range(8):
        a.add_patch(Rectangle((gx+j*gs,gy+i*gs),gs,gs,fill=False,ec='k',lw=0.5))
a.add_patch(Rectangle((gx+4*gs,gy+3*gs),gs,gs,fc='k',ec='k'))
a.add_patch(Rectangle((gx+5*gs,gy+3*gs),gs,gs,fc='0.55',ec='k'))
a.text(gx+56,gy-14,'n×n',ha='center',fontsize=15)
a.text(gx+56,gy+8*gs+20,'угловой срез',ha='center',fontsize=13,style='italic')

box(a,'4.1',280,86,100,52); box(a,'4.2',280,166,100,52)
box(a,'4',470,126,100,52)
line(a,[(gx+8*gs,112),(250,112),(250,112),(280,112)])
line(a,[(250,112),(250,192),(280,192)])
line(a,[(380,112),(430,112),(430,152),(470,152)])
line(a,[(380,192),(430,192),(430,152)],arrow=False)
a.plot(430,152,'ko',ms=4)
a.text(392,80,'до 5 пиков',fontsize=12,style='italic')
a.text(392,232,'6 признаков',fontsize=12,style='italic')

# матрица токенов по дальности (стек)
for i in range(5):
    a.add_patch(Rectangle((700+i*6,96+i*6),110,42,fill=False,ec='k',lw=1.2))
a.text(700+4*6+55,96+4*6+21,'токен',ha='center',va='center',fontsize=14)
line(a,[(570,152),(700,152)])
a.text(760,240,'матрица токенов\nпо дальности',ha='center',fontsize=13,style='italic')
a.annotate('',xy=(760,300),xytext=(760,205),arrowprops=dict(arrowstyle='-|>',color='k',lw=1.2,mutation_scale=13))
a.text(778,265,'R',fontsize=14)
cap(a,'Фиг. 2',W,440); plt.savefig(OUT+'Способ_фиг2.png',dpi=130,bbox_inches='tight',facecolor='w'); plt.close()

# ============ ФИГ. 3 — объёмный примитив АМ (детализация операции 3а) ============
W,H=1000,560; f,a=newfig(W,H)
a.text(40,40,'Данные одного такта в памяти графического процессора (ось дальности R):',fontsize=13)
# грубый проход: крупные участки
y0=70
a.add_patch(Rectangle((40,y0),880,44,fill=False,ec='k',lw=1.2))
for i in range(4):
    a.add_patch(Rectangle((40+i*220,y0),220,44,fill=False,ec='k',lw=1.6))
    a.text(40+i*220+110,y0+22,'16×16×256',ha='center',va='center',fontsize=13)
a.text(950,y0+22,'3а.1',fontsize=15,va='center')
a.text(40,y0-8,'грубый проход крупным шагом',fontsize=12,style='italic')

# отбор top-N
y1=180
box(a,'3а.2',430,y1,120,46,15)
line(a,[(150,y0+44),(150,y1+23),(430,y1+23)])
line(a,[(700,y0+44),(700,y1+23)],arrow=False)
a.plot(700,y1+23,'ko',ms=4)
a.text(180,y1+16,'отбор ограниченного числа наиболее ярких выбросов (top-N)',fontsize=12,style='italic')

# тонкий добор: те же участки мелким шагом
y2=280
a.add_patch(Rectangle((40,y2),880,44,fill=False,ec='k',lw=1.2))
for i in range(4):
    a.add_patch(Rectangle((40+220*0+i*14,y2),14,44,fill=False,ec='k',lw=1.4))
for k,x0 in [(0,100),(1,620)]:
    for i in range(8):
        a.add_patch(Rectangle((x0+i*14,y2),14,44,fc='0.85',ec='k',lw=1.2))
    a.text(x0+56,y2+62,'16×16×16',ha='center',fontsize=12)
a.text(950,y2+22,'3а.3',fontsize=15,va='center')
a.text(40,y2-8,'тонкий добор тех же участков мелким шагом (повторное 3D-БПФ)',fontsize=12,style='italic')
line(a,[(490,y1+46),(490,y2)])

box(a,'4',450,400,100,52)
line(a,[(500,324),(500,400)])
a.text(560,382,'тот же структурный токен',fontsize=13,style='italic')
cap(a,'Фиг. 3',W,520); plt.savefig(OUT+'Способ_фиг3.png',dpi=130,bbox_inches='tight',facecolor='w'); plt.close()

# ============ ФИГ. 4 — причинностный арбитр (детализация операции 6) ============
W,H=980,480; f,a=newfig(W,H)
# ось дальности
a.annotate('',xy=(900,240),xytext=(80,240),arrowprops=dict(arrowstyle='-|>',color='k',lw=1.4,mutation_scale=16))
a.text(910,246,'R',fontsize=16)
# отклики
def resp(x,h,lab,fill):
    a.add_patch(Rectangle((x-9,240-h),18,h,fc='k' if fill else 'w',ec='k',lw=1.4))
    a.text(x,240-h-14,lab,ha='center',fontsize=14)
resp(220,120,'И',True)
for i,x in enumerate([340,440,560,700]):
    resp(x,80-i*8,'Л',False)
a.text(220,268,'истинная цель\n(ближний край)',ha='center',fontsize=13,style='italic')
a.text(510,300,'ложные отметки ретранслятора: всегда ДАЛЬШЕ (τ ≥ 0)',ha='center',fontsize=13,style='italic')
# скобка задержки
a.annotate('',xy=(700,200),xytext=(220,200),arrowprops=dict(arrowstyle='<|-|>',color='k',lw=1.1,mutation_scale=12))
a.text(460,192,'c·τ / 2',ha='center',fontsize=13)
# блок арбитра
box(a,'6',440,360,100,52)
line(a,[(220,252),(220,386),(440,386)])
line(a,[(760,240),(760,386),(540,386)])
a.text(560,352,'согласование с текущим кодом (9.3)',fontsize=12,style='italic')
a.text(150,352,'передний край',fontsize=12,style='italic')
cap(a,'Фиг. 4',W,455); plt.savefig(OUT+'Способ_фиг4.png',dpi=130,bbox_inches='tight',facecolor='w'); plt.close()
print('фиг.2-4 готовы')
