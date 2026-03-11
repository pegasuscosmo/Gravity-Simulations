import os, sys, time
print("EXECUTING:", os.path.abspath(__file__))
print("PYTHON:", sys.executable)
print("STAMP:", time.strftime("%Y-%m-%d %H:%M:%S"))
import taichi as ti
import math
import time
import numpy as np
import matplotlib.pyplot as plt
ti.init(arch=ti.gpu,default_fp=ti.f32)
pi=math.pi
tau=2*pi

#Parameters

#ring size
ringRad=140000.0
innerRingRad=66900.0

#physics
n=100000
G=6.6743e-20
m=5.68e26
μ=m*G

#j2
J2 = 0.01629
Req = 60268.0   # km

#dt
T=tau*innerRingRad*math.sqrt(innerRingRad/μ)
dt=T/20

moons=[[133584,4.398,4.95e15], #pan
       [136505,3.770,7.7e13], #daphnis
       [137670,1.065,6.7e16], #atlas
       [139380,4.398,1.59e17], #prometheus
       [141700,5.655,1.37e17], #pandora
       [185539,0.45,3.749e19], #mimas
       [1221870,2.4,1.3452e23], #titan
       [238037,4.16,1.0802e20]] #enceladus
        #r,phase,mass

#collisions
thetaBinsNum=32
radialBinsNum=256

k=3
tauTarget=0.4
dtc=60
substeps=int(dt/dtc)+1

dper=0.25
rr=ringRad*(1+dper)
iRr=innerRingRad*(1-dper)
totalA=pi*rr*rr-pi*iRr*iRr

e=0.3 #coefficient of restitution
sigmaeff=4*tauTarget*totalA/n

#graph
nbins=64
figW=7 #inches for some reason
figH=2
tsize=10
#visualizer
windowSize=1000

#field init
posField=ti.Vector.field(2,dtype=ti.f32,shape=n)
velField=ti.Vector.field(2,dtype=ti.f32,shape=n)
boolField=ti.field(dtype=bool, shape=n)

moonN=len(moons)
moonPosField=ti.Vector.field(2,dtype=ti.f32,shape=moonN)
moonMuField=ti.field(dtype=ti.f32,shape=moonN)
moonPhaseField=ti.field(dtype=ti.f32,shape=moonN)
moonRField=ti.field(dtype=ti.f32,shape=moonN)

#moon fields init
for m in range(0,moonN):
    moon=moons[m]
    moonRField[m]=moon[0]
    moonPhaseField[m]=moon[1]
    moonMuField[m]=moon[2]*G

@ti.kernel
def init():
    for i in range(n):
        r=0.0
        x=0.0
        y=0.0
        boolField[i]=True
        # sample until radius is safely away from 0
        while r < innerRingRad or r>ringRad:
            x=(ti.random(ti.f32)*2*ringRad -ringRad)
            y=(ti.random(ti.f32)*2*ringRad -ringRad)
            r=ti.sqrt(x**2 + y**2)

        posField[i]=ti.Vector([x,y])

        v=ti.sqrt(μ/r) # circular speed
        disp = 0.005 * v  # ~0.1% velocity dispersion
        vx=-y/r*v # perpendicular unit tangent
        vy=x/r*v
        vx+=(ti.random(ti.f32)-0.5)*disp
        vy+=(ti.random(ti.f32)-0.5)*disp
        velField[i] = ti.Vector([vx, vy])
        velField[i]= ti.Vector([vx, vy])

@ti.kernel
def driftMoons():
    for i in range(moonN):
            r=moonRField[i]
            theta=moonPhaseField[i]
            theta=theta%tau
            meanMotion=ti.sqrt(μ/(r*r*r))
            theta=theta+meanMotion*dt
            moonPosField[i]=[r*ti.cos(theta),r*ti.sin(theta)]
            moonPhaseField[i]=theta


@ti.kernel
def drift():
    for i in range(n):
        if not boolField[i]:
            continue
        r0vec=posField[i] #initial pos
        v0vec=velField[i] #initial vel

        r0mag=r0vec.norm() #scalar form, also distance
        v0mag2=v0vec.norm_sqr()

        rv=r0vec.dot(v0vec)

        a=1.0/(2.0/r0mag - v0mag2/μ) #semi major axis

        evector=((v0mag2 -μ/r0mag)*r0vec-rv*v0vec)/μ #eccentricity vector
        e=evector.norm() #eccentricity scalar
        if e<5e-2:
            # circular orbit shortcut: rotate position by dtheta = n*dt
            n0 = ti.sqrt(μ / (r0mag * r0mag * r0mag))
            dtheta = n0 * dt
            c = ti.cos(dtheta)
            s = ti.sin(dtheta)
            rNew = ti.Vector([c * r0vec[0] - s * r0vec[1],
                            s * r0vec[0] + c * r0vec[1]])
            # v tangent with magnitude sqrt(mu/r)
            vmag = ti.sqrt(μ / r0mag)
            vNew = ti.Vector([-rNew[1] / r0mag * vmag, rNew[0] / r0mag * vmag])
            posField[i] = rNew
            velField[i] = vNew
            continue

        cosE=(1- r0mag/a)/e #eccentric anomaly
        cosE = ti.max(-1.0, ti.min(1.0, cosE))
        sinE=rv/(e*ti.sqrt(μ*a))
        E0=ti.atan2(sinE,cosE)

        M=E0-e*ti.sin(E0) #mean anomaly
        meanMotion=ti.sqrt(μ/a**3)
        M=M+meanMotion*dt #advance mean anomaly
        M=M%tau

        E=0.0
        if e<0.8:
            E=M #initial guess
        else:
            E=pi
        #iterate 3-6 times times to find E and dE
        for k in range(6):
            fE=E-e*ti.sin(E)-M
            fEinv=1-e*ti.cos(E)
            E=E- fE/fEinv #converges on actual answer
        dE=E-E0 #difference in E
        r=a*(1-e*ti.cos(E)) #new radius

        #find the f and g fucntions
        f=1.0-(a/r0mag)*(1.0-ti.cos(dE))
        g=dt-(1.0/meanMotion)*(dE-ti.sin(dE))
        rNew=f*r0vec+g*v0vec

        fdot=-(ti.sqrt(μ*a)/(r*r0mag))*ti.sin(dE)
        gdot=1.0-(a/r)*(1.0-ti.cos(dE))
        vNew=fdot*r0vec+gdot*v0vec

        #update fields
        posField[i]=rNew
        velField[i]=vNew

@ti.kernel
def kick():
    for i in range(n):
        if not boolField[i]:
            continue
        pos=posField[i]
        a=ti.Vector([0.0,0.0])
        for m in range(moonN):
            mPos=moonPosField[m]
            d=mPos-pos
            r=(d).norm()+1e-3
            if r<100:
                posField[i]=[1e10,1e10]
                boolField[i]=False
                continue
            a+=d*moonMuField[m]/(r*r*r)

        #j2
        r=pos.norm()
        a+=-3*J2/2 *μ*(Req*Req/ti.pow(r,5))*pos

        timestep=dt/2
        velField[i]+=timestep*a

rBinThickness=(rr-iRr)/radialBinsNum
tBinRadians=tau/thetaBinsNum

bIRr=rr-rBinThickness
print(rr,bIRr,rBinThickness,tBinRadians)
maxA=rr*rr*tBinRadians/2 - bIRr*bIRr*tBinRadians/2
maxD=maxA/totalA *n
print(maxD)
maxN=math.floor(10*maxD)

binField=ti.field(dtype=ti.int32,shape=(thetaBinsNum,radialBinsNum,maxN))
binOccupiedField=ti.field(dtype=ti.int32,shape=(thetaBinsNum,radialBinsNum))
binTotalVelField=ti.field(dtype=ti.f32,shape=(thetaBinsNum,radialBinsNum,4)) #xv,yv,xv2,yv2
binUsedParticle=ti.field(dtype=bool,shape=(thetaBinsNum,radialBinsNum,maxN))
mp=5

col=ti.field(dtype=ti.int32,shape=())
@ti.kernel
def collide():
    col[None]=0
    binOccupiedField.fill(0)
    binTotalVelField.fill(0)
    for i in range(n):
        if not boolField[i]:
            continue
        pos=posField[i]
        vel=velField[i]
        x=pos[0]
        y=pos[1]
        r=ti.sqrt(x*x+y*y)
        if r<iRr or r>=rr:
            continue
        t=ti.atan2(y,x)
        if t<0:
            t+=tau
        thetaIdx=ti.cast(t/tBinRadians,ti.i32)
        thetaIdx=ti.min(thetaBinsNum-1,ti.max(0, thetaIdx))
        radiusIdx=ti.cast(ti.floor((r-iRr)/rBinThickness),ti.int32)
        targetIdx=binOccupiedField[thetaIdx,radiusIdx]
        if targetIdx>=maxN:
            continue
        binField[thetaIdx,radiusIdx,targetIdx]=i
        binOccupiedField[thetaIdx,radiusIdx]+=1
        binTotalVelField[thetaIdx,radiusIdx,0]+=vel[0]
        binTotalVelField[thetaIdx,radiusIdx,1]+=vel[1]
        binTotalVelField[thetaIdx,radiusIdx,2]+=vel[0]*vel[0]
        binTotalVelField[thetaIdx,radiusIdx,3]+=vel[1]*vel[1]
    for steps in range(substeps):
        for thetaIdx in range(thetaBinsNum):
            for radIdx in range(radialBinsNum):
                thetaIdxInt=ti.cast(thetaIdx,ti.int32)
                radIdxInt=ti.cast(radIdx,ti.int32)
                N=binOccupiedField[thetaIdxInt,radIdxInt]
                if N<2:
                    continue
                vxMean=binTotalVelField[thetaIdxInt,radIdxInt,0]/N
                vyMean=binTotalVelField[thetaIdxInt,radIdxInt,1]/N
                vx2Mean=binTotalVelField[thetaIdxInt,radIdxInt,2]/N
                vy2Mean=binTotalVelField[thetaIdxInt,radIdxInt,3]/N
                xDisp=vx2Mean-vxMean*vxMean
                yDisp=vy2Mean-vyMean*vyMean
                vDisp=ti.sqrt(xDisp+yDisp)
                innerRad=radIdx*rBinThickness+iRr
                rc=innerRad+rBinThickness/2
                vShear=3/2 *rBinThickness*ti.sqrt(μ/(ti.pow(rc,3)))
                vMax=vShear+k*vDisp
                if vMax<=0:
                    continue
                area= tBinRadians*(ti.pow((innerRad+rBinThickness),2)-ti.pow(innerRad,2))/2
                Vproxy=ti.min(0.1,N/area *sigmaeff*vMax*dtc)
                maxpairs=0.05*N
                pairs=ti.cast(0.5*N*ti.min(Vproxy,0.5),ti.i32)
                pairs=ti.max(pairs,0)
                pairs=ti.min(maxpairs,pairs)
                for i in range(pairs):
                    particleACellIdx=ti.cast(ti.random(ti.f32)*N,ti.int32)
                    particleBCellIdx=ti.cast(ti.random(ti.f32)*N,ti.int32)
                    aIdx=binField[thetaIdx, radIdx, particleACellIdx]
                    bIdx=binField[thetaIdx, radIdx, particleBCellIdx]
                    if aIdx==bIdx:
                        continue
                    if binUsedParticle[thetaIdxInt,radIdxInt,aIdx] or binUsedParticle[thetaIdxInt,radIdxInt,bIdx]:
                        continue
                    binUsedParticle[thetaIdxInt,radIdxInt,aIdx]=True
                    binUsedParticle[thetaIdxInt,radIdxInt,bIdx]=True
                    aVel=velField[aIdx]
                    bVel=velField[bIdx]
                    relativeVelVector=aVel-bVel
                    relativeVelScalar=ti.sqrt(ti.pow(relativeVelVector[0],2)+ti.pow(relativeVelVector[1],2))
                    P=ti.min(1,relativeVelScalar/vMax)
                    if P>ti.random(ti.f32):
                        col[None]+=1
                        rvec=posField[aIdx]-posField[bIdx]
                        dist=rvec.norm() +1e-6
                        rhat=rvec / dist
                        vrel_n=relativeVelVector.dot(rhat)
                        if vrel_n>=0:
                            continue
                        J =-(1+e)*vrel_n/(1/mp +1/mp)
                        velField[aIdx]+=(J/mp)*rhat
                        velField[bIdx]-=(J/mp)*rhat

    



posDraw = ti.Vector.field(2, dtype=ti.f32, shape=n)
moonDraw = ti.Vector.field(2, dtype=ti.f32, shape=moonN)
@ti.kernel
def to_screen(scale: ti.f32):
    for i in range(n):
        if boolField[i]:
            posDraw[i] = posField[i] * scale + ti.Vector([0.5, 0.5])
        else:
            posDraw[i] = ti.Vector([ti.math.nan, ti.math.nan])
    for i in range(moonN):
        moonDraw[i]=moonPosField[i]*scale+ti.Vector([0.5,0.5])

window = ti.ui.Window("Orbit", (windowSize, windowSize))
canvas = window.get_canvas()

r_edges = np.linspace(iRr, rr, nbins + 1)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
annulus_area = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)

plt.ion()
fig, ax = plt.subplots(figsize=(figW, figH))
line, = ax.plot([], [], lw=1)
ax.set_title("Ring density compared to flat distribution",fontsize=tsize)
ax.set_xlabel("Radius",fontsize=tsize)
ax.set_ylabel("Relative surface density",fontsize=tsize)
ax.grid(True)
ax.tick_params(axis='both', labelsize=tsize)

a=0.25
c="red"
for m in range(0,moonN):
    r=moons[m][0]
    if r>rr:
        continue
    ax.axvline(x=r,color=c,alpha=a,lw=1, ls="--")

r_edges = np.linspace(iRr, rr, nbins + 1)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
annulus_area = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)

A=pi*((ringRad*1.00)**2 - (innerRingRad*1.00)**2)
avgd = n/A

def update_radial_density():
    pos = posField.to_numpy()                 # (n,2)
    alive = boolField.to_numpy().astype(bool) # (n,)
    pos = pos[alive]
    if pos.size == 0:
        return

    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)

    counts, _ = np.histogram(r, bins=r_edges)
    sigma = counts / annulus_area
    sigmaRelative=sigma/avgd
    
    line.set_data(r_centers, sigmaRelative)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=False)  # we control both axes anyway
    ax.set_ylim(0.0, max(1.0, counts.max() * 1.1))
    ax.set_ylim(sigmaRelative.min()/1.1, sigmaRelative.max()*1.1)

    fig.canvas.draw_idle()
    plt.pause(0.001)


@ti.kernel
def countLiving() -> ti.int32:
    alive=0
    for i in range(n):
        if boolField[i]:
            if posField[i].norm()<rr:
                alive+=1
    return ti.cast(alive, ti.int32)

# Precompile
init()
driftMoons()
drift()
scale=1.0/(2*ringRad*1.25)

to_screen(scale)
ti.sync()

frame=0
while window.running:
    driftMoons()
    kick()
    drift()
    kick()
    collide()
    if frame%10:
        to_screen(scale)
        canvas.set_background_color((0,0,0))
        canvas.circles(posDraw, radius=0.0010, color=(0.4,0.4,0.4))  # radius is normalized units
        canvas.circles(moonDraw,radius=0.005,color=(0.7,0.7,0.7))
        window.show()
    
    if frame % 60 == 0:
        update_radial_density()
        print(frame,countLiving(),col[None])
    frame += 1