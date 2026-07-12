import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

W,H = 1000, 920
fig,ax = plt.subplots(figsize=(W/100,H/100),dpi=130)
ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')

BW,BH = 90,50           # main box
SW,SH = 80,42           # sub box

boxes={}
def box(name,x,y,w=BW,h=BH,fs=17):
    ax.add_patch(Rectangle((x,y),w,h,fill=False,ec='black',lw=1.6))
    ax.text(x+w/2,y+h/2,name,ha='center',va='center',fontsize=fs)
    boxes[name]=(x,y,w,h)

def line(pts,arrow=True):
    for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
        ax.plot([x1,x2],[y1,y2],color='black',lw=1.2,solid_capstyle='butt')
    if arrow:
        (x1,y1),(x2,y2)=pts[-2],pts[-1]
        dx,dy=x2-x1,y2-y1
        n=(dx**2+dy**2)**.5
        ax.annotate('',xy=(x2,y2),xytext=(x2-dx/n*1,y2-dy/n*1),
                    arrowprops=dict(arrowstyle='-|>',color='black',lw=1.2,mutation_scale=13))

def dot(x,y): ax.plot(x,y,'o',color='black',ms=4)

# ---- главная колонка (x=430) ----
MX=430
ys={'1':40,'2':120,'3':200,'4':320,'5':440,'6':520,'8':600,'9':680,'10':800}
for k,y in ys.items(): box(k,MX,y)

# ---- под-операции справа (x=640 / 780) ----
box('3.1',640,196,SW,SH,14); box('3.2',760,196,SW,SH,14)
box('4.1',640,300,SW,SH,14); box('4.2',640,352,SW,SH,14)
box('9.1',640,676,SW,SH,14); box('9.2',760,676,SW,SH,14); box('9.3',880,676,SW,SH,14)

# ---- слева: 3а (АМ-фронтенд) и 7 (добор) ----
box('3а',180,260,BW,BH,15)
box('7',180,440,BW,BH)

# ================= СВЯЗИ (только 90°) =================
c=MX+BW/2   # 475  центр главной колонки
# 1 -> 2 -> 3
line([(c,90),(c,120)])
line([(c,170),(c,200)])
# 3 -> 4 -> 5 -> 6
line([(c,250),(c,320)])
line([(c,370),(c,440)])
line([(c,490),(c,520)])
# 6 -> 8 -> 9
line([(c,570),(c,600)])
line([(c,650),(c,680)])
# 9 -> 10
line([(c,730),(c,800)])
# 10 -> 1  (обратная связь, слева x=60)
line([(MX,825),(60,825),(60,65),(MX,65)])
ax.text(66,450,'следующий такт',rotation=90,fontsize=11,style='italic',va='center')

# 3 -> 3.1 -> 3.2   (декомпозиция)
line([(MX+BW,217),(640,217)])
line([(720,217),(760,217)])
# 4 -> 4.1 ; 4 -> 4.2
line([(MX+BW,345),(600,345)],arrow=False)
line([(600,345),(600,321),(640,321)])
line([(600,345),(600,373),(640,373)])
dot(600,345)
# 9 -> 9.1 -> 9.2 -> 9.3
line([(MX+BW,697),(640,697)])
line([(720,697),(760,697)])
line([(840,697),(880,697)])
# 9.3 -> 6  (подтверждение код-корреляцией)
line([(920,676),(920,560),(MX+BW,560)])
ax.text(700,552,'подтверждение кодом',fontsize=11,style='italic')

# 5 -> 7  (трудный кандидат) ; 7 -> 6 (уточнённые угол/дальность)
line([(MX,455),(270,455)])
ax.text(360,448,'трудный кандидат',fontsize=11,style='italic',ha='center')
line([(225,490),(225,545),(MX,545)])

# 3а -> 4  (альтернативный фронтенд, тот же токен)
line([(270,285),(350,285),(350,335),(MX,335)])
ax.text(292,280,'АМ-фронтенд',fontsize=11,style='italic')

ax.text(W/2,890,'Фиг. 1',ha='center',fontsize=17,fontweight='bold')
plt.savefig('/sessions/eager-exciting-hypatia/mnt/Patent/Способ.png',dpi=130,bbox_inches='tight',facecolor='white')
print('saved')
