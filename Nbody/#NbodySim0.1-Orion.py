import pygame
from pygame.locals import *
import numpy as np
import random
import math
import time

# screen
sW=1800 #screen width
sH=1000 #screen height
zoom=1
offset=np.array([0,0])

#physics
bodyCount=2000
k=16 #quadtree splitting threshold
θ=1 #quadtree travelling threshold
G=1 #gravitational constant
dt=1/200
ep=10
maxD=32 #max depth of quadtree

#performance tracking
calcs=0
pureCalcs=0
quadsTravelled=0
pointInsertCalled=0

#other
frame=0





#particle arrays
posArr=np.zeros((bodyCount,2))
velArr=np.zeros((bodyCount,2))
massArr=np.zeros((bodyCount))
tempPosArr=posArr.copy()
tempVelArr=velArr.copy()

for n in range(0,bodyCount):
    while True:
        posArr[n]=[random.randint(0,sW)-sW/2,random.randint(0,sW)-sW/2]
        dist=math.sqrt((posArr[n,0])**2+(posArr[n,1])**2)
        if dist<sW/2:
            break
    massArr[n]=1000

#quadtree class
class quadRect():
    def __init__(self,x,y,w,depth,m=0,mpos=[0.0,0.0]):
        self.d=depth
        self.x=x
        self.y=y
        self.w=w
        self.childList=[]
        self.bodyCount=0
        self.bodyList=[]
        self.m=m
        self.mPos=np.array(mpos, dtype=np.float64)
        self.comPosX=0
        self.comPosY=0
    def split(self):
        UL=quadRect(self.x-self.w/2,self.y+self.w/2,self.w/2,self.d+1)
        UR=quadRect(self.x+self.w/2,self.y+self.w/2,self.w/2,self.d+1)
        DL=quadRect(self.x-self.w/2,self.y-self.w/2,self.w/2,self.d+1)
        DR=quadRect(self.x+self.w/2,self.y-self.w/2,self.w/2,self.d+1)
        for n in self.bodyList:
            if posArr[n,0]>self.x:
                if posArr[n,1]>self.y:
                    UR.insertPoint(n)
                else:
                    DR.insertPoint(n)
            else:
                if posArr[n,1]>self.y:
                    UL.insertPoint(n)
                else:
                    DL.insertPoint(n)
        self.childList=[UL,UR,DL,DR]
        self.bodyList=[]
        self.bodyCount=0
    def insertPoint(self,pointIndex):
        global pointInsertCalled
        pointInsertCalled+=1
        pos=posArr[pointIndex]
        self.m+=massArr[pointIndex]
        self.mPos+=posArr[pointIndex]*massArr[pointIndex]
        if len(self.childList)>0:
            if pos[0]>self.x:
                if pos[1]>self.y:
                    self.childList[1].insertPoint(pointIndex)
                else:
                    self.childList[3].insertPoint(pointIndex)
            else:
                if pos[1]>self.y:
                    self.childList[0].insertPoint(pointIndex)
                else:
                    self.childList[2].insertPoint(pointIndex)
        else:
            self.bodyCount+=1
            self.bodyList.append(pointIndex)
            if self.bodyCount>k and self.d<maxD:
                self.split()
    def findCOM(self):
        if self.m>0:
            self.comPosX=self.mPos[0]/self.m
            self.comPosY=self.mPos[1]/self.m
            if len(self.childList)>0:
                for n in range(0,4):
                    self.childList[n].findCOM()
    def delete(self):
        if len(self.childList)>0:
            self.childList[0].delete()
            self.childList[1].delete()
            self.childList[2].delete()
            self.childList[3].delete()
        del self
#other functions
def travelQuad(node,body):
    global velArr,posArr, pureCalcs, calcs, quadsTravelled
    bx = posArr[body, 0]
    by = posArr[body, 1]
    stack = [node]
    while stack:
        node = stack.pop()   
        dx=node.comPosX - bx
        dy=node.comPosY - by
        dist2=dx**2 + dy**2 + ep**2
        dist=math.sqrt(dist2)
        pureCalcs+=1
        quadsTravelled+=1
        if node.m==0:
            continue
        if dist==ep:
            continue
        if 2*node.w/dist<θ or len(node.childList)==0:
            calcs+=1
            accel=dt*G*node.m/(dist2*dist) #GM/dist**2, dist**3 comes from GM/dist**2 *ndist/dist = (GM*ndist)/(dist*dist**2), take ndist out for 2 dimensions, ndist * (GM/dist**3)
            velArr[body]+=[accel*dx,accel*dy] #ndist * (GM/dist**3) ^^^
            continue
        else:
            stack.extend(node.childList)
def calc():
    global tempPosArr,tempVelArr,posArr, frame
    posArr+=dt*velArr/2
    tempPosArr=posArr.copy()
    tempVelArr=velArr.copy()
    frame+=1
    jitter=[0,0]
    if frame%10==0:
        jitter=np.array([random.uniform(-1,1), random.uniform(-1,1)])
    root=quadRect(jitter[0], jitter[1], 2**11, 0)
    for n in range(0,bodyCount):
        root.insertPoint(n)
    root.findCOM()
    for n in range(0,bodyCount):
        travelQuad(root,n)
    posArr+=dt*velArr/2
    root.delete()
pixelArr=np.zeros((sW,sH,3),dtype=np.uint8)
def draw():
    screen.fill((0,0,0))
    pixelArr.fill(0)
    tempPosArr=posArr.copy()
    tempPosArr[:,0]=tempPosArr[:,0]*zoom+offset[0]*zoom +sW/2
    tempPosArr[:,1]=tempPosArr[:,1]*zoom+offset[1]*zoom +sH/2
    tempPosArr = tempPosArr[tempPosArr[:, 0] < sW]
    tempPosArr = tempPosArr[tempPosArr[:, 0] > 0]
    tempPosArr = tempPosArr[tempPosArr[:, 1] < sH]
    tempPosArr = tempPosArr[tempPosArr[:, 1] > 0]
    tempPosArr=np.int_(tempPosArr)
    pixelArr[tempPosArr[:,0], tempPosArr[:,1]] = [100, 100, 100]
    pygame.surfarray.blit_array(screen, pixelArr)
def naiveCalc():
    global posArr,calcs
    posArr+=velArr/2
    for n in range(0,bodyCount):
        diffArr=posArr-posArr[n]
        distArr=np.sqrt(np.sum(diffArr**2,axis=1))
        distArr[n]=100
        accelArr=dt*G*massArr/distArr**2
        xAccelMultArr=diffArr[:,0]/distArr
        yAccelMultArr=diffArr[:,1]/distArr
        xAccel=np.sum(accelArr*xAccelMultArr)
        yAccel=np.sum(accelArr*yAccelMultArr)
        velArr[n,0]+=xAccel*dt
        velArr[n,1]+=yAccel*dt
        calcs+=bodyCount
    posArr+=velArr/2

#initializing pygame
pygame.init()
screen = pygame.display.set_mode((sW,sH),pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.SCALED)
pygame.display.set_caption("nbody")
_font_cache = {}
def get_font(size):
    f = _font_cache.get(size)
    if f is None:
        f = pygame.font.SysFont("calibri", size)
        _font_cache[size] = f
    return f

def show_text(msg, x, y, color, size):
    fontobj = get_font(size)
    msgobj = fontobj.render(msg, False, color)
    screen.blit(msgobj,(x,y))
clock=pygame.time.Clock()



offsetDir=np.array([0,0])
while True:
    break
    for event in pygame.event.get():
                if event.type==QUIT:
                    pygame.quit()
                    exit()
                if event.type==KEYDOWN:
                    key=event.key
                    if key==K_ESCAPE:
                        pygame.quit()
                        exit()
                    if key==K_a:
                        offsetDir[0]=1
                    if key==K_d:
                        offsetDir[0]=-1
                    if key==K_s:
                        offsetDir[1]=-1
                    if key==K_w:
                        offsetDir[1]=1
                if event.type==KEYUP:
                    key=event.key
                    if key==K_a and offsetDir[0]==1:
                        offsetDir[0]=0
                    if key==K_d and offsetDir[0]==-1:
                        offsetDir[0]=0
                    if key==K_s and offsetDir[1]==-1:
                        offsetDir[1]=0
                    if key==K_w and offsetDir[1]==1:
                        offsetDir[1]=0
                if event.type==MOUSEWHEEL:
                        if event.y>0:
                            zoom=zoom*(2**(1/3))
                        elif event.y<0:
                            zoom=max(0.00006103515625,zoom/2**(1/3))

    calcs=0
    pureCalcs=0
    quadsTravelled=0
    pointInsertCalled=0

    offset+=offsetDir*10

    calc()
    draw()
    fps=str(clock.get_fps()*1000//10/100)
    show_text(fps+" FPS",0,0,(255,255,0),30)
    print(calcs/bodyCount**2, ",", pureCalcs/bodyCount**2, ",", quadsTravelled, ",", pointInsertCalled,",",fps)
    pygame.display.update()
    clock.tick(60)

print("\n",bodyCount,"\n")
start=time.time()
calc()
end=time.time()
print("\nBH")
timer=end-start
print(timer,1/timer,timer/bodyCount)

start=time.time()
calc()
end=time.time()
print("\nNaive")
timer=end-start
print(timer,1/timer,timer/bodyCount)