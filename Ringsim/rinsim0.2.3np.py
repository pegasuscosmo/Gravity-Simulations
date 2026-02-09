import numpy as np
import pygame
from pygame.locals import *
import random 
import math
import numba
from numba import njit

bodyCount=0;sW=0;sH=0;Whalf=0;Hhalf=0;G=0;dt=0;tau=2*math.pi
rings=[];Main=[];majorBodyCreationList=[];majorBodyList=[]
randomize=False
objectArr=np.zeros((bodyCount,4))
pixelArr=np.zeros((sW,sH,3),dtype=np.uint8)

x=0
def generate():
    global bodyCount, sW,sH,Whalf,Hhalf,G,dt,rings,Main,majorBodyCreationList, majorBodyList, objectArr, pixelArr,x
    if x==0:
        #basic/screen parameters
        bodyCount=50000
        sW=1800
        sH=1000

        Whalf=sW/2
        Hhalf=sH/2

        #time affecting parameters
        G=100
        dt=1/200

        #randomizable parameters
        rings=[[50,200,1]] #id,od,%
        #ignore vvvvvvvvvvvvvvv
        Main=  [Whalf,Hhalf,0,0,5972,6.3,()] #x,y,xv,yv,mass,radius, orbit color
        majorBodyCreationList=[] #starting x, starting y, periapsis dist, mass, radius, orbit color

        #ignore
        Main[4]*=G
    majorBodyList=[Main]
    x=1
    objectArr=np.zeros((bodyCount,4))
    pixelArr=np.zeros((sW,sH,3),dtype=np.uint8)
        
    generateSystem()


def generateSystem():
    global objectArr, bodyCount, Whalf, Hhalf, sW, sH, G, randomize, Main, majorBodyList, majorBodyCreationList, rings
    Main[0]=Whalf; Main[1]=Hhalf; Main[2]=0; Main[3]=0
    if randomize:
        Main=[Whalf,Hhalf,0,0,random.randint(1000,10000)*G,random.randint(40,80)]
        majorBodyList=[Main]
        majorBodyCreationList=[]
        majorNum=random.randint(1,2)
        
        for n in range(0,majorNum):
            while True:
                x=random.randint(0,int(Whalf))-Whalf/2
                y=random.randint(0,int(Hhalf))-Hhalf/2
                dist=int((x**2+y**2) **0.5)
                if dist>Main[5]*1.26:
                    break
            periapsis=random.randint(int(Main[5]*1.25),int(dist))
            mass=random.randint(int(Main[4]/G/400),int(Main[4]/G/70))
            r=int(math.log(mass))
            color=(random.randint(50,255),random.randint(50,255),random.randint(50,255))
            majorBodyCreationList.append([x,y,periapsis,mass,r,color])
        rings=[]
        p=0
        while True:
            outerR=0
            innerR=0
            while outerR<=innerR+10:
                innerR=random.randint(int(Main[5]*1.25),int(sW/4))
                outerR=random.randint(int(innerR),int(sW/3))
            if p>=85:
                per=(100-p)/100
                p=100
                break
            else:
                per=random.randint(15,int(100-p))/100
            p+=per*100
            rings.append([innerR,outerR,per])
        print(majorBodyCreationList,rings)

    index = 0
    for innerRingRadius, ringRadius, percent in rings:
        count = int(bodyCount * percent)
        ringRadius10k = int(10000 * ringRadius)

        for _ in range(count):
            while True:
                x = random.randint(-ringRadius10k, ringRadius10k) / 10000
                y = random.randint(-ringRadius10k, ringRadius10k) / 10000
                dist = (x*x + y*y) ** 0.5
                if dist <= ringRadius and dist >= innerRingRadius:
                    break

            v = (Main[4] / dist) ** 0.5
            objectArr[index] = [x + Whalf, y + Hhalf, v * (y / dist), -v * (x / dist)]
            index += 1

            if index >= bodyCount:
                break
        if index >= bodyCount:
            break

    for n in range(0,len(majorBodyCreationList)):
        x=majorBodyCreationList[n][0]
        y=majorBodyCreationList[n][1]
        periapsisDist=majorBodyCreationList[n][2]
        m=majorBodyCreationList[n][3]
        r=majorBodyCreationList[n][4]
        c=majorBodyCreationList[n][5]
        majorBodyList.append([x+Whalf,y+Hhalf,0,0,m*G,r,c])
        dist=(x**2+y**2)**0.5 #central body is always at (0,0)
        e=(dist-periapsisDist)/(dist+periapsisDist)
        v=(Main[4]*(1-e)/dist)**0.5
        majorBodyList[n+1][2]=v*(y)/dist #xvel
        majorBodyList[n+1][3]=v*(-x)/dist #yvel

def updatePos():
    global objectArr, majorBodyList
    #filter out of bounds
    objectArr = objectArr[objectArr[:, 0] < 10*sW]
    objectArr = objectArr[objectArr[:, 0] > -9*sW]
    objectArr = objectArr[objectArr[:, 1] < 10*sH]
    objectArr = objectArr[objectArr[:, 1] > -9*sH]
    for n in range(0,len(majorBodyList)):
        #particles
        r=majorBodyList[n][5]
        distAxisArr=[majorBodyList[n][0],majorBodyList[n][1]]-objectArr[:,:2]
        dist2Arr=np.sum(distAxisArr**2,axis=1,keepdims=True)
        accelArr=majorBodyList[n][4]/dist2Arr
        accelArr=accelArr*[1,1]
        distFactorArr=distAxisArr/dist2Arr**0.5
        accelArr*=distFactorArr
        accelArr*=dttw
        objectArr[:,2:]+=accelArr
        #filter inside major body
        posArr=objectArr[:,:2]
        distArr=np.linalg.norm(posArr-[majorBodyList[n][0],majorBodyList[n][1]],axis=1)
        distMaskArr=distArr>r
        objectArr=objectArr[distMaskArr]
    objectArr[:,:2]+=objectArr[:,2:]*dttw

    #majors
    majorDeathQueue=[]
    for n in range(0,len(majorBodyList)):
        xAccel=0
        yAccel=0
        for m in range(0,len(majorBodyList)):
            a=majorBodyList[n]
            b=majorBodyList[m]
            if m==n:
                continue
            xDist=b[0]-a[0]
            yDist=b[1]-a[1]
            dist2=xDist**2 + yDist**2
            if dist2==0:
                continue
            dist=dist2**0.5
            if m==0 and dist<Main[5]:
                majorDeathQueue.append(n)
            accel=b[4]/dist2 *dttw
            xAccel+=accel* xDist/dist
            yAccel+=accel* yDist/dist
        majorBodyList[n][2]+=xAccel 
        majorBodyList[n][3]+=yAccel
    for n in range(0,len(majorBodyList)):
        majorBodyList[n][0]+=majorBodyList[n][2]*dttw
        majorBodyList[n][1]+=majorBodyList[n][3]*dttw
    for n in majorDeathQueue:
        majorBodyList.pop(n)
    draw()
def trajectory(x,y,xv,yv,cx,cy,cm,color):
    xDist=cx-x
    yDist=cy-y
    dist2=xDist**2 + yDist**2
    dist=dist2**0.5
    if dist<Main[5]:
        return "end"
    accel=cm/dist2 *dt
    xAccel=accel* xDist/dist
    yAccel=accel* yDist/dist
    xv+=xAccel
    yv+=yAccel
    x+=xv*dt
    y+=yv*dt
    drawx=x*zoom+Whalf+zoomXOffset
    drawy=y*zoom+Hhalf+zoomYOffset
    screen.set_at((int(drawx), int(drawy)), color)
    return [x,y,xv,yv,cx,cy]

def draw(): 
    global fps, pixelArr, zoom, zoomXOffset,zoomYOffset,fps,tabbedIn
    fps=str(clock.get_fps()*1000//10/100)
    if tabbedIn:
        #particles
        pixelArr.fill(0)
        posArr=objectArr[:,:2].copy()
        posArr[:,:1]=posArr[:,:1]*zoom+Whalf+zoomXOffset
        posArr[:,1:2]=posArr[:,1:2]*zoom+Hhalf+zoomYOffset
        posArr=np.int_(posArr)
        posArr = posArr[posArr[:, 0] < sW]
        posArr = posArr[posArr[:, 0] > 0]
        posArr = posArr[posArr[:, 1] < sH]
        posArr = posArr[posArr[:, 1] > 0]
        pixelArr[posArr[:,0], posArr[:,1]] = [100, 100, 100]
        pygame.surfarray.blit_array(screen, pixelArr)
        # trajectory drawing
        if newBody!=[]:
            newBodyX=newBody[0]
            newBodyY=newBody[1]
            newBodyRad=newBody[2]
            newBodyMass=newBody[3]
            if newBodyMass>10:
                col=(255,255,0)
            else:
                col=(255,0,0)
            pygame.draw.circle(screen,(255,255,0),(newBodyX,newBodyY),newBodyRad*zoom)
            show_text(str(newBodyMass),newBodyX-math.log10(newBodyMass)*10,newBodyY-30-(newBodyRad),col,30)
            x=(newBodyX - Whalf - zoomXOffset) / zoom
            y=(newBodyY - Hhalf - zoomYOffset) / zoom
            xv=newBody[4]
            yv=newBody[5]
            mainX=Main[0]
            mainY=Main[1]
            steps=10000
            for i in range(0,int(steps/G/dt)):
                t=trajectory(x,y,xv,yv,mainX,mainY,Main[4],(100,100,0))
                if t=="end":
                    break
                x=t[0]; y=t[1]; xv=t[2]; yv=t[3]
                mainX=t[4]; mainY=t[5]
                #mainX+=Main[2]*dt
                #mainY+=Main[3]*dt
        steps=10000
        for n in majorBodyList:
            if n==Main:
                continue
            x=n[0]
            y=n[1]
            xv=n[2]
            yv=n[3]
            mainX=Main[0]
            mainY=Main[1]
            color=n[6]
            for n in range(0,steps):
                t=trajectory(x,y,xv,yv,mainX,mainY,Main[4],color)
                if t=="end":
                    break
                x=t[0]; y=t[1]; xv=t[2]; yv=t[3]
                mainX=t[4]; mainY=t[5]
                #mainX+=Main[2]*dt
                #mainY+=Main[3]*dt
        #majors
        for n in majorBodyList:
            x=int((n[0])*zoom+Whalf+zoomXOffset)
            y=int((n[1])*zoom+Hhalf+zoomYOffset)
            r=n[5]
            if r*zoom>1:
                pygame.draw.circle(screen,(250,250,250),(x,y),r*zoom)
            else:
                screen.set_at((x, y), (250,250,250))

        #other screen stuff
        currentBodyCount=str(len(objectArr)//100/10) +"k"
        pygame.draw.rect(screen,(0,0,0),(sW-135,0,140,60))
        show_text("FPS: "+fps,sW-120,10,(255,255,0),20)
        show_text(currentBodyCount,sW-120,30,(255,255,0),20)
        show_text("Starting conditions:    Rings: "+str(rings)+"      Majors: "+str([Main[4]//G,Main[5]])+","+str(majorBodyCreationList), 10, sH-10,(255,255,0), 10)
        mBLint=[]
        for n in majorBodyList:
            x=int(n[0]-Whalf)
            y=int(n[1]-Hhalf)
            xv=n[2]
            yv=n[3]
            v=int((xv**2+yv**2)**0.5)
            m=int(n[4]/G)
            mBLint.append([x,y,v,m])
        show_text("Current majors [x,y,v,m]: "+str(mBLint),10,sH-30,(255,255,0),15)
        #update screen
        pygame.display.flip()



def mainLoop():
    global dttw, zoomXOffset, zoomYOffset, fps, tabbedIn, zoom, randomize, majorBodyList, newBody
    restart=0
    sqrt2Inv=1/ 2**0.5
    xMoveInput=0
    yMoveInput=0
    xoffset=-Whalf
    yoffset=-Hhalf
    zoom=1
    offsetSpeed=0.5
    tw=1
    tabbedIn=True
    x=0
    fps="0"
    dttw=dt*tw
    newBodyMass=1
    newBodyRadius=1
    newBodyXv=0
    newBodyYv=0
    newBodyX=0
    newBodyY=0
    newBody=[]
    x,y=0,0
    newBodyMassInput=0; newBodyRadInput=0
    while True:
        for event in pygame.event.get():
                if event.type==QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.WINDOWFOCUSLOST:
                    tabbedIn=False
                if event.type == pygame.WINDOWFOCUSGAINED:
                    tabbedIn=True
                if event.type==KEYDOWN:
                    if event.key==K_ESCAPE:
                        pygame.quit()
                        exit()
                    # restart and randomize
                    if event.key==K_f:
                        restart=1
                    if event.key==K_r:
                        restart=2
                    # camera and timewarp
                    if event.key==K_l:
                        tw*=2
                    if event.key==K_j:
                        tw*=1/2
                    if event.key==K_k:
                        tw=1
                    if event.key==K_w:
                        yMoveInput=1
                    if event.key==K_s:
                        yMoveInput=-1
                    if event.key==K_a:
                        xMoveInput=1
                    if event.key==K_d:
                        xMoveInput=-1
                    if event.key==K_LSHIFT or event.key==K_RSHIFT:
                        offsetSpeed=2
                        newBodyXv=(newBodyX-mouseX)*offsetSpeed
                        newBodyYv=(newBodyY-mouseY)*offsetSpeed
                    if event.key==K_LCTRL:
                        offsetSpeed=0.125
                        newBodyXv=(newBodyX-mouseX)*offsetSpeed
                        newBodyYv=(newBodyY-mouseY)*offsetSpeed
                    if event.key==K_UP:
                        newBodyMassInput=1
                    if event.key==K_DOWN:
                        newBodyMassInput=-1
                    if event.key==K_RIGHT:
                        newBodyRadInput=1
                    if event.key==K_LEFT:
                        newBodyRadInput=-1
                    if event.key==K_BACKSLASH:
                        newBodyMass*=2
                    if event.key==K_BACKSPACE:
                        newBodyMass*=0.5
                        newBodyMass=int(newBodyMass)
                if event.type==KEYUP:
                    if event.key==K_w and yMoveInput==1:
                        yMoveInput=0
                    if event.key==K_s and yMoveInput==-1:
                        yMoveInput=0
                    if event.key==K_a and xMoveInput==1:
                        xMoveInput=0
                    if event.key==K_d and xMoveInput==-1:
                        xMoveInput=0
                    if event.key==K_LSHIFT or event.key==K_RSHIFT:
                        offsetSpeed=0.5
                        newBodyXv=(newBodyX-mouseX)*offsetSpeed
                        newBodyYv=(newBodyY-mouseY)*offsetSpeed
                    if event.key==K_LCTRL:
                        offsetSpeed=0.5
                        newBodyXv=(newBodyX-mouseX)*offsetSpeed
                        newBodyYv=(newBodyY-mouseY)*offsetSpeed
                    if event.key==K_UP:
                        newBodyMassInput=0
                    if event.key==K_DOWN:
                        newBodyMassInput=0
                    if event.key==K_RIGHT:
                        newBodyRadInput=0
                    if event.key==K_LEFT:
                        newBodyRadInput=0
                if event.type==MOUSEWHEEL:
                        if event.y>0:
                            zoom=zoom*(2**(1/3))
                        elif event.y<0:
                            zoom=max(0.00006103515625,zoom/2**(1/3))
                if event.type==MOUSEMOTION:
                    newBodyXv=(newBodyX-event.pos[0])*offsetSpeed
                    newBodyYv=(newBodyY-event.pos[1])*offsetSpeed
                    mouseX=event.pos[0]
                    mouseY=event.pos[1]
                if event.type==MOUSEBUTTONDOWN:
                    x=event.pos[0]
                    y=event.pos[1]
                    newBodyX=x
                    newBodyY=y
                    newBody=[newBodyX,newBodyY,newBodyRadius,newBodyMass,newBodyXv,newBodyYv]
                if event.type==MOUSEBUTTONUP:
                    x=0
                    y=0
                    if newBodyMass>10:
                        newBodyColor=(random.randint(50,255),random.randint(50,255),random.randint(50,255))
                        majorBodyList.append([(newBodyX-zoomXOffset-Whalf)/zoom,(newBodyY-zoomYOffset-Hhalf)/zoom,newBodyXv,newBodyYv,newBodyMass*G,newBodyRadius,newBodyColor])
                    newBodyMass=1
                    newBodyRadius=1
                    newBodyXv=0
                    newBodyYv=0
                    newBodyX=0
                    newBodyY=0
                    newBody=[]
        if x!=0 and y!=0:
            newBodyMass+=newBodyMassInput*10*offsetSpeed**2
            newBodyMass=max(newBodyMass,1)
            newBodyRadius+=newBodyRadInput
            
            newBody=[newBodyX,newBodyY,newBodyRadius,newBodyMass,newBodyXv,newBodyYv]
        if xMoveInput!=0 and yMoveInput!=0:
            dirMult=sqrt2Inv
        else:
            dirMult=1
        dirMult*=offsetSpeed
        zoomeffect=zoom**0.5 /10 /dirMult
        xoffset+=xMoveInput/zoomeffect
        yoffset+=yMoveInput/zoomeffect
        zoomXOffset=xoffset*zoom
        zoomYOffset=yoffset*zoom
        dttw=dt*tw
        screen.fill((0,0,0))
        updatePos()
        clock.tick(240)
        if restart==1:
            restart=0
            randomize=True
            runSim()
        if restart==2:
            restart=0
            randomize=False
            runSim()


def runSim():
    generate()

generate()
#initializing pygame
pygame.init()
screen = pygame.display.set_mode((sW,sH),pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.SCALED)
pygame.display.set_caption("ring sim")
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
runSim()
mainLoop()