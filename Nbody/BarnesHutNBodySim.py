import taichi as ti
from taichi.algorithms import parallel_sort
ti.reset()
ti.init(arch=ti.cpu,default_fp=ti.f32)

#simulation parameters
#   particles
n=100000
m=10
#   physics
G=1
dt=0.005
theta=1 #acceptance angle, larger -> more performance, less accuracy
eps=2e0 #smoothing, recommended >1e-1
#   generation
preset=3
k=2
k2=2
k3=0.4
spinMult=-2
drawScale=100
#       generation manual
#preset 0: square generation, k=random velocity amount, k2=inward velocity amount (scaled by distance), no spin
#preset 1: circle generation, k=random velocity amount, k2=inward velocity amount (scaled by distance), has spin
#preset 2: galaxy generation, k=radial distribution power (<2 advised), k2=arm amount, k3=distance scaled spin multiplier, has spin
#preset 3: galaxy merger generation, k=radial distribution power (<2 advised), k2=arm amount, k3=distance between galaxy multiplier (k3=1 -> 3*drawScale), has spin
#spin mult - flat multiplier on spin
#drawScale - generation bounding box size
#k,k2,k3 - generation parameters, defined above

#   visual
#       window
windowSize=1000
#       colors
pc=0.1 #brightness sensitivity
pb=3 #bloom
alphaC=1.2 #zoom bright scale    ^ =zoom in ^ brightness  v = zoom in v brightness
gammaC=1 #contrast
gammaB=1 #contrast of bloom
c1=ti.Vector([0.8,0.8,0.5]) #high density
c2=ti.Vector([0.7,0.4,0.6]) #mid density
c3=ti.Vector([0.3,0.2,0.7])#low density
midC=0.3


#fields
posField=ti.Vector.field(2,dtype=ti.f32,shape=n) #x,y
velField=ti.Vector.field(2,dtype=ti.f32,shape=n) #xv,yv

drawScaleInv=1/drawScale

@ti.kernel
def init(preset: ti.int32, k: ti.f32, k2: ti.int32, k3: ti.f32):
    if preset==0: #square init
        for i in range(n):
            x=ti.random()*drawScale
            y=ti.random()*drawScale
            posField[i]=ti.Vector([x,y])
            vx=(ti.random()-0.5)*k
            vy=(ti.random()-0.5)*k
            velField[i]=ti.Vector([vx,vy]) - k2*(posField[i]-ti.Vector([drawScale/2,drawScale/2]))
    elif preset==1: #circle init
        for i in range(n):
            r=ti.cast(drawScale+1,ti.f32)
            x=0.0
            y=0.0
            d=ti.Vector([0.0,0.0])
            while r>drawScale/2 and r>0:
                x=ti.random()*drawScale
                y=ti.random()*drawScale
                d=ti.Vector([x,y])-ti.Vector([drawScale/2,drawScale/2])
                r=d.norm()
            posField[i]=ti.Vector([x,y])
            vx=(ti.random()-0.5)*k
            vy=(ti.random()-0.5)*k
            velField[i]=ti.Vector([-d[1],d[0]])/r * ti.sqrt(G*m*n*r/(drawScale/2)**2)*spinMult + ti.Vector([vx,vy])  - k2*(posField[i]-ti.Vector([drawScale/2,drawScale/2]))
    elif preset==2: #galaxy init
        for i in range(n):
            g=0
            gcenter=ti.Vector([drawScale/2+g*drawScale*1.5,drawScale/2])
            r=ti.cast(drawScale+1,ti.f32)
            x=0.0
            y=0.0
            d=ti.Vector([0.0,0.0])
            while r>drawScale/2 and r>0:
                rad=ti.random()**k *drawScale/2
                theta=ti.random()*2*ti.math.pi
                theta+=1+0.5*ti.sin(k2*theta+(ti.random()-0.5)*ti.math.pi)
                x=rad*ti.cos(theta)+gcenter[0]
                y=rad*ti.sin(theta)+gcenter[1]
                d=ti.Vector([x,y])-gcenter
                r=d.norm()
            posField[i]=ti.Vector([x,y])
            if r>drawScale/40:
                velField[i]=ti.Vector([-d[1],d[0]])/r * ti.sqrt(G*m*n/2.0*r/(drawScale/2.0)**2)*spinMult * ti.pow(drawScale/2/r,k3) * ti.pow(2,r/(drawScale/2))
            else:
                velField[i]=ti.Vector([ti.random()-0.5,ti.random()-0.5])*m*n*G/10000
    elif preset==3: #galaxy merger init
        for i in range(n):
            g=k3
            if i%2:
                g=-g
            gcenter=ti.Vector([drawScale/2+g*drawScale*1.5,drawScale/2])
            r=ti.cast(drawScale+1,ti.f32)
            x=0.0
            y=0.0
            d=ti.Vector([0.0,0.0])
            while r>drawScale/4 and r>0:
                rad=ti.random()**k *drawScale/4
                theta=ti.random()*2*ti.math.pi
                theta+=1+0.5*ti.sin(k2*theta+(ti.random()-0.5)*ti.math.pi)
                x=rad*ti.cos(theta)+gcenter[0]
                y=rad*ti.sin(theta)+gcenter[1]
                d=ti.Vector([x,y])-gcenter
                r=d.norm()
            posField[i]=ti.Vector([x,y])
            velField[i]=ti.Vector([-d[1],d[0]])/r * ti.sqrt(G*m*n/2*r/(drawScale/4)**2)*spinMult/2 +ti.Vector([0,g*0])
            



#chatgpt, adapted to taichi
@ti.func
def splitBy1(x: ti.uint64) -> ti.uint64:
    x=ti.cast(x,ti.uint64)
    x &= ti.u64(0xFFFFFFFF)  # ensure 32-bit
    x = (x | (x << 16)) & ti.u64(0x0000FFFF0000FFFF)
    x = (x | (x << 8))  & ti.u64(0x00FF00FF00FF00FF)
    x = (x | (x << 4))  & ti.u64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << 2))  & ti.u64(0x3333333333333333)
    x = (x | (x << 1))  & ti.u64(0x5555555555555555)
    return x
#end cite

#code fields and variables
chunkBits=8 #8 seems to be the best
chunkNum=64//chunkBits

codeField=ti.field(dtype=ti.u64,shape=n)
sortedCodeField=ti.field(dtype=ti.u64,shape=n)
pointerField=ti.field(dtype=ti.int32,shape=n)
sortedPointerField=ti.field(dtype=ti.int32,shape=n)

#chatgpt
@ti.func
def maskBits(code,k):
    mask = (1 << chunkBits) - 1
    return ti.cast((code >> k) & mask,ti.uint64)
#end cite

scale=2**31-1

@ti.kernel
def createCodes(): #create morton codes   
    maxPosX=0
    maxPosY=0
    for i in range(n): #find max positions
        ti.atomic_max(maxPosX,posField[i][0])
        ti.atomic_max(maxPosY,posField[i][1])
    maxScaleX=scale/maxPosX #create scaling factor to go from f32->int32 while 
    maxScaleY=scale/maxPosY #maximizing how many bits are used in the resulting i32
    for i in range(n):
        x=posField[i][0]
        y=posField[i][1]
        x*=maxScaleX #scale positions to int32
        y*=maxScaleY
        x=ti.cast(x,ti.int32) #cast to i32 first to avoid f32->u64 cast bugginess
        y=ti.cast(y,ti.int32)
        x=ti.cast(x,ti.uint64) #cast to u64 for morton coding
        y=ti.cast(y,ti.uint64)
        codeField[i]=(splitBy1(x)|(splitBy1(y)<<1)) #create morton code
        pointerField[i]=i

# morton coding quick explanation
# morton codes are a 1d representation of n-dimensional positions that preserve locality
# example -
# x: 100101, y: 011001
# xSplitBy1: 1000100010, ySplitBy1: 001010000010
# shift y left 1, then or x and y
# code: 100101100011
# in effect this does-
# x=ABCDEF, y=GHIJKL, code=AGBHCIDJEKFL
# this creates this z order curve:
# 0  1  4  5
# 2  3  6  7
# 8  9  12 13
# 10 11 14 15
# which is a space filling curve that preserves locality, hence why morton codes themselves also preserve locality
# this implicitly creates a quadtree, which is necessary for Barnes Hut


@ti.func
def binarySearch(L: ti.int32,R: ti.int32,k: ti.int32) -> ti.int32: #find split index
    low=L
    high=R-1
    while low<high:
        mid=(low+high+1)//2
        if findK(L,mid)>k:
            low=mid
        else:
            high=mid-1

    return low

#chatgpt
@ti.func
def msb(x: ti.u64) -> ti.i32:
    idx=0
    if (x>>32)!=0:
        x=x>>32
        idx+=32
    if (x>> 16) != 0:
        x=x>>16
        idx+=16
    if (x>>8)!=0:
        x=x>>8
        idx+=8
    if (x>>4)!=0:
        x=x>>4
        idx+=4
    if (x>>2)!=0:
        x=x>>2
        idx+=2
    if (x>>1)!=0:
        idx+=1
    return idx
#end cite

@ti.func
def findK(L: ti.int32,R: ti.int32) -> ti.int32: #find first differing bit
    i=0
    if R<0 or R>=n:
        i=-1
    else:
        xor=codeField[L]^codeField[R]
        if xor==0:
            i=64+findKTieBreaker(L,R)
        else:
            i=63-msb(xor)
    return i

@ti.func
def findKTieBreaker(L: ti.int32,R: ti.int32) -> ti.int32: #find differing bit between the indices
    i=0
    if R<0 or R>=n:
        i=-1
    else:
        xor=L^R
        if xor==0:
            i=64
        else:
            i=63-msb(xor)
    return i

#tree node fields
nodeN=2*n-1
nodeRangeField=ti.Vector.field(2,dtype=ti.int32, shape=nodeN,)
nodeParentField=ti.field(dtype=ti.int32, shape=nodeN)
nodeChildField=ti.Vector.field(2,dtype=ti.int32, shape=nodeN)

nodeMassField=ti.field(dtype=ti.f32,shape=nodeN)
nodeCOMField=ti.Vector.field(2,dtype=ti.f32,shape=nodeN)
nodeABoundField=ti.Vector.field(2,dtype=ti.f32,shape=nodeN) #min x,y
nodeBBoundField=ti.Vector.field(2,dtype=ti.f32,shape=nodeN) #max x,y

@ti.kernel
def createTree(): #create the binary LVBH tree
    for i in range(2*n-1):#reset fields
        nodeParentField[i]=-1
    for i in range(n): #initialize leaves
        nodeRangeField[i] = ti.Vector([i, i])
        nodeABoundField[i]=posField[sortedPointerField[i]]
        nodeBBoundField[i]=posField[sortedPointerField[i]]
        nodeCOMField[i]=posField[sortedPointerField[i]]
        nodeMassField[i]=m
    for i in range(n-1):
        #taichi adapted chatgpt code (literally magic)
        dp=findK(i,i+1)
        dn=findK(i,i-1)
        d=1
        if dp-dn<0:
            d=-1  
        deltaMin=findK(i,i-d)
        lMax=1
        while findK(i,i+lMax*d) > deltaMin:
            lMax*=2
        l = 0
        t = lMax
        while t > 1:
            t //= 2
            if findK(i, i + (l + t) * d) > deltaMin:
                l += t
        j=i+l*d
        L=ti.min(i,j)
        R=ti.max(i,j)
        L=ti.max(0,ti.min(i,j))
        R=ti.min(n-1,ti.max(i,j))
        idx=i+n
        gamma=findK(L,R)
        split=binarySearch(L,R,gamma)
        left=0
        right=0
        if L==split:
            left=split
        else:
            left=split+n
        if R==split+1:
            right=split+1
        else:
            right=split+1+n
        #end cite

        nodeRangeField[idx]=ti.Vector([L,R])
        nodeChildField[idx]=ti.Vector([left,right])
        nodeParentField[left]=idx
        nodeParentField[right]=idx
#the tree created looks like this (somehow):
# 0,  1,  2,  ...  ,  n-1  ,  n  ,  n+1,  n+2,  ...  ,  2*n-2
# leaf nodes                 root   internal nodes


#fields for mass and bound allocation
set=ti.field(dtype=ti.int32,shape=n)
setLen=ti.field(dtype=ti.int32,shape=())
nextSet=ti.field(dtype=ti.int32,shape=n)
nextSetLen=ti.field(dtype=ti.int32,shape=())
readyCount=ti.field(ti.int32,nodeN)

@ti.kernel
def massBoundAllocate() -> ti.int32: #allocate masses, COMs, and bounds
    l=setLen[None]
    nextSetLen[None]=0
    for i in range(l):
        node=set[i]
        if node>=n:
            left=nodeChildField[node][0]
            right=nodeChildField[node][1]
            leftM=nodeMassField[left]
            rightM=nodeMassField[right]
            leftCOM=nodeCOMField[left]
            rightCOM=nodeCOMField[right]
            nodeM=leftM+rightM
            nodeCOM=(leftCOM*leftM + rightCOM*rightM)/nodeM

            nodeMassField[node]=nodeM
            nodeCOMField[node]=nodeCOM

            leftABound=nodeABoundField[left]
            leftBBound=nodeBBoundField[left]
            rightABound=nodeABoundField[right]
            rightBBound=nodeBBoundField[right]

            aBound=ti.Vector([ti.min(leftABound[0],rightABound[0]),ti.min(leftABound[1],rightABound[1])])
            bBound=ti.Vector([ti.max(leftBBound[0],rightBBound[0]),ti.max(leftBBound[1],rightBBound[1])])

            nodeABoundField[node]=aBound
            nodeBBoundField[node]=bBound
        p=nodeParentField[node]
        if p!=-1:
            old=ti.atomic_add(readyCount[p],1)
            if old==1:
                oldLen=ti.atomic_add(nextSetLen[None],1)
                nextSet[oldLen]=p
    return nextSetLen[None]

@ti.kernel
def massBoundAllocateInit(): #initialize fields and variables for allocation
    setLen[None]=0
    nextSetLen[None]=n
    for i in range(nodeN):
        readyCount[i]=0
        if i<n:
            set[i]=0
            nextSet[i]=i

@ti.kernel
def swap(): #swap set and nextSet
    for i in range(n):
        set[i]=nextSet[i]
    temp=setLen[None]
    setLen[None]=nextSetLen[None]
    nextSetLen[None]=temp

def massBoundAllocateController():
    global set, nextSet, setLen, nextSetLen
    massBoundAllocateInit()
    length=n
    while length>0:
        swap()
        length=massBoundAllocate()

#quick explanation of the allocation logic
# nodes require their childrens masses, COM, and bounds to be calculated, 
# which in turn need their own children and so on
# so, we start with the leaves (already calculated), then ping their parents
# when pinged (+1 to their readycount) and their ready count hits 2 (all children ready),
# the node is added to nextSet, and after all nodes in the current set have finished, the
# nodes in nextSet are transferred to the current set to be calculated.
# this continues until nextSet is empty


#stacks for each particle
maxStackDepth=128
stack=ti.field(dtype=ti.int32,shape=(n,maxStackDepth))
stackTop=ti.field(dtype=ti.int32,shape=n)
@ti.kernel
def forceAccumulate(): #accelerate particles, velocity verlet integration
    for i in range(n):
        posField[sortedPointerField[i]]+=velField[sortedPointerField[i]]*dt/2
        stackTop[i]=0
    for i in range(n):
        stack[i,0]=n
        stackTop[i]=1
        pos=posField[sortedPointerField[i]]
        accelVec=ti.Vector([0.0,0.0])
        while stackTop[i]>0:
            #setup
            stackTop[i]-=1
            node=stack[i,stackTop[i]]
            
            w=nodeBBoundField[node]-nodeABoundField[node]
            maxW=ti.max(w[0],w[1])
            nodePos=nodeCOMField[node]
            d=nodePos-pos
            r=d.norm()+eps
            L=nodeRangeField[node][0]
            R=nodeRangeField[node][1]
            containI=(L<=i and i<=R)
            
            #BH gravity
            if node<n:
                if node!=i:
                    if r>eps:
                        accel=G*nodeMassField[node]/(r*r)
                        accelVec+=accel*d/r
            else:
                if r>eps:
                    if maxW/r>theta or containI:
                        if stackTop[i]<maxStackDepth-2 and nodeChildField[node][1]!=-1:
                            stack[i,stackTop[i]]=nodeChildField[node][0]
                            stackTop[i]+=1
                            stack[i,stackTop[i]]=nodeChildField[node][1]
                            stackTop[i]+=1

                    else:
                        accel=G*nodeMassField[node]/(r*r)
                        accelVec+=accel*d/r
        velField[sortedPointerField[i]]+=accelVec*dt
    for i in range(n):    
        posField[sortedPointerField[i]]+=velField[sortedPointerField[i]]*dt/2

#quick explanation time again!
# each particle has their own stack, which starts with the root node
# if the nodes largest side length divided by the distance from the 
# particle to the node's COM is less than the parameter theta, the node is accepted. 
# this is effectively seeing if the node is contained within a certain angle (theta) from the particle
# when a node is accepted, it is treated like its own particle, and normal newtonian gravity is applied.
# if the node isn't accepted, it adds its children to the stack. 
# This in effect creates a depth first search for nodes that satisfy s/d<theta


window = ti.ui.Window("Barnes Hut Sim", (windowSize, windowSize))
canvas = window.get_canvas()

posDraw = ti.Vector.field(2, dtype=ti.f32, shape=n)

pixelField=ti.field(dtype=ti.f32,shape=(windowSize,windowSize))
imgField=ti.Vector.field(3,dtype=ti.f32,shape=(windowSize,windowSize))

@ti.kernel
def to_screen(scale: ti.f32, zoom: ti.f32, offset: ti.types.vector(2,ti.f32)):
    for i in range(n):
        p=posField[i]-ti.Vector([1/scale/2,1/scale/2])+offset
        scaleP=p*scale/zoom
        posDraw[i]=(scaleP+ti.Vector([0.5,0.5]))*windowSize
#taichi windows use [0,1] as their coordinates, so to draw them, positions must be scaled first

@ti.kernel
def drawDensity(zoomInv: ti.f32):
    for i in range(n):
        pos=posDraw[i]
        x=ti.cast(pos[0],ti.int32)
        y=ti.cast(pos[1],ti.int32)
        if x<windowSize and x>0 and y>0 and y<windowSize:
            a=1
            ti.atomic_add(pixelField[x,y],a)
            if x+1<windowSize:
                ti.atomic_add(pixelField[x+1,y],0.25*a)
                if y+1<windowSize:
                    ti.atomic_add(pixelField[x+1,y+1],0.05*a)
                if y-1>0:
                    ti.atomic_add(pixelField[x+1,y-1],0.05*a)
            if x-1>0:
                ti.atomic_add(pixelField[x-1,y],0.25*a)
                if y+1<windowSize:
                    ti.atomic_add(pixelField[x-1,y+1],0.05*a)
                if y-1>0:
                    ti.atomic_add(pixelField[x-1,y-1],0.05*a)
            if y+1<windowSize:
                ti.atomic_add(pixelField[x,y+1],0.25*a)
            if y-1>0:
                ti.atomic_add(pixelField[x,y-1],0.25*a)
    for i1,i2 in pixelField:
        if pixelField[i1,i2]!=0:
            v=pixelField[i1,i2]*ti.pow(zoomInv,alphaC)
            v2=pixelField[i1,i2]*ti.pow(zoomInv,2)
            b=ti.pow(1-ti.exp(-pc*v),gammaC)
            bb=ti.pow(1-ti.exp(-pb*v2),gammaB)
            t=(b-midC)/(1-midC)
            c=c2*(1-t)+c1*t #smoothing logic and above variables modified from chat gpt
            if b<midC:
                t=b/midC
                c=c3*(1-t)+c2*t
            imgField[i1,i2]=c*bb
        else:
            imgField[i1,i2]=ti.Vector([0,0,0])

def sim():
    zoom=1.2
    zoomInv=1/zoom
    offset=ti.Vector([0.0,0.0])
    while window.running:
        window.get_events()
        #zoom
        if window.is_pressed('i'):
            zoom *= 0.95
        if window.is_pressed('k'):
            zoom *= 1.05
        #movement
        if window.is_pressed('w'):
            offset[1]-=zoom*(drawScale/100)
        if window.is_pressed('s'):
            offset[1]+=zoom*(drawScale/100)
        if window.is_pressed('a'):
            offset[0]+=zoom*(drawScale/100)
        if window.is_pressed('d'):
            offset[0]-=zoom*(drawScale/100)
        zoomInv=1/zoom
        #physics logic
        createCodes()
        parallel_sort(codeField,pointerField)
        sortedPointerField.copy_from(pointerField)
        createTree()
        massBoundAllocateController()
        forceAccumulate()
        #draw logic
        pixelField.fill(0)
        to_screen(drawScaleInv,zoom, offset)
        drawDensity(zoomInv)
        canvas.set_image(imgField)
        window.show()

init(preset,k,k2,k3)
sim()
