using CUDA
using GLMakie
import AcceleratedKernels as AK


n=262_144 #divisible by 256
a0=35
mparticle=4

m=1
c=1
dt=0.05
#SPH
h=0.1
rho0=0.1 #target density
k=0.001 #density->pressure
mu=0.1 #viscosity

accreted=0


bhPos=CUDA.zeros(Float32, 2,3)
bhVel=CUDA.zeros(Float32, 2,3)

pos=CUDA.randn(Float32, n,3)
vel=CUDA.zeros(Float32, n,3)

posMtrx=CUDA.zeros(UInt64, n,3)
codes=CUDA.zeros(UInt64, n)
pointers=CuArray(UInt32.(1:n))
reversePointers=CUDA.zeros(UInt32, n)


#GEN
function bhInit(bhPos,bhVel, m, a0)
    bhPos[1,1]=a0/2
    bhPos[2,1]=-a0/2
    M=2m
    η=m^2/M^2
    x=M/a0
    u=x
    for _ in 1:30
        A1=(1+3η)*(-u) +(2+η)*2x
        A2=(-η*(3-4η)*u^2 
            +η/2*(13-4η)*u*x 
            -3/4*(12+29η)*x^2)

        u=x*(1-A1-A2)
    end
    s=sqrt(u)/2
    #s=sqrt(M/a0)/2
    bhVel[1,2]=s
    bhVel[2,2]=-s
    nothing
end

function posInitialize!(pos,bhPos)
    i=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    mDiskIR=7
    mDiskW=2
    mDiskH=0.25
    cDiskIR=50
    cDiskW=100
    cDiskH=15
    b=rem(i,3)+1
    if b<=2
        bx=bhPos[b,1]
        by=bhPos[b,2]
        bz=bhPos[b,3]
        t=rand(Float32)*2pi
        r=rand(Float32)*mDiskW+mDiskIR
        z=(r-mDiskIR)*mDiskH*randn(Float32)
        x=cos(t)*r
        y=sin(t)*r
        pos[i,1]=x+bx
        pos[i,2]=y+by
        pos[i,3]=z+bz
    else
        t=rand(Float32)*2pi
        r=rand(Float32)*cDiskW+cDiskIR
        z=cDiskH*randn(Float32)
        x=cos(t)*r
        y=sin(t)*r
        pos[i,1]=x
        pos[i,2]=y
        pos[i,3]=z
    end
    nothing
end

function velInitialize!(pos,vel,m,bhPos,bhVel)
    i=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    b=rem(i,3)+1
    if b<=2
        bx=bhPos[b,1]
        by=bhPos[b,2]
        bz=bhPos[b,3]
        bvx=bhVel[b,1]
        bvy=bhVel[b,2]
        bvz=bhVel[b,3]
        x=pos[i,1]
        y=pos[i,2]
        z=pos[i,3]
        dx=bx-x
        dy=by-y
        dz=bz-z
        r2=dx*dx+dy*dy
        r=sqrt(r2)
        s=sqrt(m*r)/(r-2m)
        ux=dx/r
        uy=dy/r
        uz=dz/r
        vx=s*uy
        vy=-s*ux
        vz=-0.02s*uz
        vel[i,1]=vx+bvx
        vel[i,2]=vy+bvy
        vel[i,3]=vz+bvz
    else
        x=pos[i,1]
        y=pos[i,2]
        z=pos[i,3]
        t=atan(x,y)
        r2=x*x+y*y
        r=sqrt(r2)
        M=2m
        s=sqrt(M*r)/(r-2M)
        ux=x/r
        uy=y/r
        uz=z/r
        vx=-s*uy
        vy=s*ux
        vz=-0.1s*uz
        vel[i,1]=vx
        vel[i,2]=vy
        vel[i,3]=vz
    end
    nothing
end

#MORTON CODING
#magic bit dilation
function splitBy3(x::UInt64)
    x &= 0x1fffff
    x = (x | (x << 32)) & UInt64(0x1f00000000ffff)
    x = (x | (x << 16)) & UInt64(0x1f0000ff0000ff)
    x = (x | (x <<  8)) & UInt64(0x100f00f00f00f00f)
    x = (x | (x <<  4)) & UInt64(0x10c30c30c30c30c3)
    x = (x | (x <<  2)) & UInt64(0x1249249249249249)
    return x
end

function morton(x,y,z)
    xs=splitBy3(x)
    ys=splitBy3(y)
    zs=splitBy3(z)
    m=xs | (ys<<1) | (zs<<2)
    return m
end

function mortonCodeGPU!(pos,codes, mins, maxs)
    i=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    dx=maxs[1]-mins[1]
    dy=maxs[2]-mins[2]
    dz=maxs[3]-mins[3]
    x1=(pos[i,1]-mins[1])/dx*Float32(2.0^21-1)
    y1=(pos[i,2]-mins[2])/dy*Float32(2.0^21-1)
    z1=(pos[i,3]-mins[3])/dz*Float32(2.0^21-1)
    x=UInt64(floor(x1))
    y=UInt64(floor(y1))
    z=UInt64(floor(z1))
    codes[i]=morton(x, y, z)
    nothing
end

function resetPointers!(pointers)
    i=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    pointers[i]=i
    nothing
end

#TREE CREATION
function binarySearch(L,R,k, codes, n) #find split index
    low=L
    high=R-1
    while low<high
        mid=fld(low+high+1,2)
        if findK(L,mid, codes, n)>k
            low=mid
        else
            high=mid-1
        end
    end
    return low
end

function findK(L,R, codes, n) #find first differing bit
    i=0
    if R<0 || R>=n
        i=-1
    else
        XOR=xor(codes[L+1],codes[R+1])
        if XOR==0
            i=64+findKTieBreaker(L,R, n)
        else
            i=leading_zeros(XOR)
        end
    end
    return i
end

function findKTieBreaker(L,R, n) #find differing bit between the indices
    i=0
    if R<0 || R>=n
        i=-1
    else
        XOR=xor(UInt64(L),UInt64(R))
        if XOR==0
            i=64
        else
            i=leading_zeros(XOR)
        end
    end
    return i
end

#tree node fields
nodeN=2*n-1
nodeRangeField=CUDA.zeros(Int32, nodeN,2)
nodeParentField=CUDA.zeros(Int32, nodeN)
nodeChildField=CUDA.zeros(Int32, nodeN,3)
nodeABoundField=CUDA.zeros(Float32, nodeN,3) #min x,y
nodeBBoundField=CUDA.zeros(Float32, nodeN,3) #max x,y
ready=CUDA.zeros(Int32, nodeN)

function createTree!(nodeRangeField, nodeParentField, nodeChildField,
                     nodeABoundField, nodeBBoundField,
                    pos, codes, n, pointers, reversePointers) #create the binary LVBH tree
    g=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    if g<=2n-1
        if g<=n   #initialize leaves
            i=g
            p=Int(pointers[i])
            nodeRangeField[i,1]=i
            nodeRangeField[i,2]=i
            nodeABoundField[i,1]=pos[p,1]
            nodeBBoundField[i,1]=pos[p,1]
            nodeABoundField[i,2]=pos[p,2]
            nodeBBoundField[i,2]=pos[p,2]
            nodeABoundField[i,3]=pos[p,3]
            nodeBBoundField[i,3]=pos[p,3]
            reversePointers[p]=i
        else
            if g==n+1
                nodeParentField[g]=0
            end
            i=g-n-1
            #karras 2012 LVBH algorithm
            dp=findK(i,i+1, codes, n)
            dn=findK(i,i-1, codes, n)
            d=1
            if dp-dn<0
                d=-1 
            end 
            deltaMin=findK(i,i-d, codes, n)
            lMax=1
            while findK(i,i+lMax*d, codes, n) > deltaMin
                lMax*=2
            end
            l = 0
            t = lMax
            while t > 1
                t=fld(t,2)
                if findK(i, i+(l+t)*d, codes, n) > deltaMin
                    l+=t
                end
            end
            j=i+l*d
            L=min(i,j)
            R=max(i,j)
            L=max(0,min(i,j))
            R=min(n-1,max(i,j))
            idx=i+n
            gamma=findK(L,R, codes, n)
            split=binarySearch(L,R,gamma, codes, n)
            left=0
            right=0
            if L==split
                left=split
            else
                left=split+n
            end
            if R==split+1
                right=split+1
            else
                right=split+1+n
            end

            nodeRangeField[idx+1,1]=L+1
            nodeRangeField[idx+1,2]=R+1
            nodeChildField[idx+1,1]=left+1
            nodeChildField[idx+1,2]=right+1
            nodeParentField[left+1]=idx+1
            nodeParentField[right+1]=idx+1
        end
    end
    nothing
end

function accumulateBounds(nodeABoundField, nodeBBoundField, nodeParentField, ready)
    i=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    p=nodeParentField[i]
    r=1
    while p!=0 && r==1
        CUDA.@atomic nodeABoundField[p,1]=min(nodeABoundField[p,1],nodeABoundField[i,1])
        CUDA.@atomic nodeABoundField[p,2]=min(nodeABoundField[p,2],nodeABoundField[i,2])
        CUDA.@atomic nodeABoundField[p,3]=min(nodeABoundField[p,3],nodeABoundField[i,3])
        CUDA.@atomic nodeBBoundField[p,1]=max(nodeBBoundField[p,1],nodeBBoundField[i,1])
        CUDA.@atomic nodeBBoundField[p,2]=max(nodeBBoundField[p,2],nodeBBoundField[i,2])
        CUDA.@atomic nodeBBoundField[p,3]=max(nodeBBoundField[p,3],nodeBBoundField[i,3])
        CUDA.threadfence()
        r=CUDA.atomic_add!(pointer(ready,p),Int32(1))
        i=p
        p=nodeParentField[i]
    end
    nothing
end

active=CUDA.zeros(Int8,n)
#GRAVITY
function gravitateParticles!(pos, bhPos, vel, n, m,  dt, active)
    i=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    if 0<i<=n
        if active[i]==1
            return
        end
        for j in 1:2
            dx=bhPos[j,1]-pos[i,1]
            dy=bhPos[j,2]-pos[i,2]
            dz=bhPos[j,3]-pos[i,3]
            r2=dx*dx+dy*dy+dz*dz
            r=sqrt(r2)
            rb=sqrt(pos[i,1]*pos[i,1]+pos[i,2]*pos[i,2])
            if r<2.1m || rb>500m
                vel[i,1]=0
                vel[i,2]=0
                vel[i,3]=0
                active[i]=1
                break
            else
                ux=dx/r
                uy=dy/r
                uz=dz/r
                a=dt*m/(r-2m)^2
                vel[i,1]+=ux*a
                vel[i,2]+=uy*a
                vel[i,3]+=uz*a
            end
        end
        pos[i,1]+=vel[i,1]*dt
        pos[i,2]+=vel[i,2]*dt
        pos[i,3]+=vel[i,3]*dt
    end
    nothing
end

function postNewtonBH(bhPos,bhVel, m,dt)
    #half step
    bhPos[1,1]+=dt*bhVel[1,1]/2
    bhPos[1,2]+=dt*bhVel[1,2]/2
    bhPos[1,3]+=dt*bhVel[1,3]/2
    bhPos[2,1]+=dt*bhVel[2,1]/2
    bhPos[2,2]+=dt*bhVel[2,2]/2
    bhPos[2,3]+=dt*bhVel[2,3]/2

    #PNE setup
    dx=bhPos[1,1]-bhPos[2,1]
    dy=bhPos[1,2]-bhPos[2,2]
    dz=bhPos[1,3]-bhPos[2,3]
    dvx=bhVel[1,1]-bhVel[2,1]
    dvy=bhVel[1,2]-bhVel[2,2]
    dvz=bhVel[1,3]-bhVel[2,3]

    r2=dx*dx+dy*dy+dz*dz
    r=sqrt(r2)
    v2=dvx*dvx+dvy*dvy+dvz*dvz
    v=sqrt(v2)

    dr=(dx*dvx+dy*dvy+dz*dvz)/r

    nx=dx/r
    ny=dy/r
    nz=dz/r
    
    M=2m
    η=m^2/M^2
    mrinv=M/r

    #Post Newton terms up to order 2.5 (5 halves)
    #PN1
    A1PN=-(1+3η)v2 +3η/2 *dr^2 +2(2+η)mrinv
    B1PN=2(2-η)
    #PN2
    A2PN=(-η*(3-4η)*v2^2 
    +η/2 * (13-4η)*v2 * mrinv 
        +3η/2 *(3-4η)*v2 * dr^2 
        +(2+25η+2η^2)*dr^2 *mrinv 
        -15η/8 *(1-3η)*dr^4 
        -3/4 *(12+29η)*(mrinv)^2)
    B2PN=(η/2 *(15+4η)*v2 
        -3η/2 *(3+2η)*dr^2 
        -1/2 *(4+41η+8η^2)*mrinv)
    #PN2.5
    A5halfPN=3v2+17/3 *mrinv
    B5halfPN=v2+3mrinv

    #calculate *n and *v terms
    #*n
    Cn=(-M/r2 *(1-A1PN-A2PN)
    +8/5 *η*M^2/r^3 * dr*A5halfPN)
    #*v
    Cv=(M/r2 *dr*(B1PN + B2PN)
    -8/5 *η *M^2/r^3 *B5halfPN)

    #relative acceleration
    aRelX=Cn*nx+Cv*dvx
    aRelY=Cn*ny+Cv*dvy
    aRelZ=Cn*nz+Cv*dvz

    #vel update
    bhVel[1,1]+=dt*aRelX/2
    bhVel[1,2]+=dt*aRelY/2
    bhVel[1,3]+=dt*aRelZ/2
    bhVel[2,1]-=dt*aRelX/2
    bhVel[2,2]-=dt*aRelY/2
    bhVel[2,3]-=dt*aRelZ/2
    #half step
    bhPos[1,1]+=dt*bhVel[1,1]/2
    bhPos[1,2]+=dt*bhVel[1,2]/2
    bhPos[1,3]+=dt*bhVel[1,3]/2
    bhPos[2,1]+=dt*bhVel[2,1]/2
    bhPos[2,2]+=dt*bhVel[2,2]/2
    bhPos[2,3]+=dt*bhVel[2,3]/2
    nothing
end

#smoothing kernel, (x-h)^2
function W(x,h)
    y=Float32((x-h)^2)
    V=Float32(2pi*h^5 /15)
    y=y/V 
    return y
end
#derivative
function dW(x,h)
    y=Float32(15(x-h)/(pi*h^5))
    return y
end
#second derivative
function d2W(x,h)
    y=Float32(15/(pi*h^5))
    return y
end

stackSize=256
traversalStack=CUDA.zeros(Int32, n,stackSize)
rhoField=CUDA.zeros(Float32, n)
pField=CUDA.zeros(Float32, n)

function getRhoP(traversalStack, k,rho0, pos,n,mparticle,h, active,rhoField,pField, pointers, nodeABoundField,nodeBBoundField,nodeChildField)
    i=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    if active[i]==1
        return
    end
    x=pos[i,1]
    y=pos[i,2]
    z=pos[i,3]
    stackTop=2
    traversalStack[i,1]=n+1
    h2=h*h
    rhoi=Float32(0.0)
    while stackTop>1
        s=traversalStack[i,stackTop-1]
        aBoundX=nodeABoundField[s,1]
        aBoundY=nodeABoundField[s,2]
        aBoundZ=nodeABoundField[s,3]
        bBoundX=nodeBBoundField[s,1]
        bBoundY=nodeBBoundField[s,2]
        bBoundZ=nodeBBoundField[s,3]
        clampX=min(bBoundX,max(aBoundX,x))
        clampY=min(bBoundY,max(aBoundY,y))
        clampZ=min(bBoundZ,max(aBoundZ,z))
        dx=clampX-x
        dy=clampY-y 
        dz=clampZ-z
        r2=dx*dx+dy*dy+dz*dz
        if r2<=h2
            if s<=n
                j=pointers[s]
                if active[j]==0
                    rhoi+=W(sqrt(r2),h)
                end
                stackTop-=1
                
            else
                if stackTop<=257
                    traversalStack[i,stackTop-1]=nodeChildField[s,1]
                end
                if stackTop<=256
                    traversalStack[i,stackTop]=nodeChildField[s,2]
                    stackTop+=1
                end
            end
        else
            stackTop-=1
        end
    end
    rhoi=rhoi*mparticle
    rhoField[i]=rhoi
    pField[i]=k*(rhoi-rho0)
    nothing
end

vF=CUDA.zeros(Float32, n)
#https://www.cs.cornell.edu/courses/cs5643/2015sp/stuff/BridsonFluidsCourseNotes_SPH_pp83-86.pdf
function navierStokes(traversalStack, pointers, pos,vel,vF,n,dt,h,mparticle, active,rhoField,pField,mu, nodeABoundField,nodeBBoundField,nodeChildField)
    i=(blockIdx().x - 1)*blockDim().x + threadIdx().x
    x=pos[i,1]
    y=pos[i,2]
    z=pos[i,3]
    stackTop=2
    traversalStack[i,1]=n+1
    h2=h*h
    p_i=pField[i]
    v_ix=vel[i,1]
    v_iy=vel[i,2]
    v_iz=vel[i,3]
    if active[i]==1
        vF[i]=sqrt(v_ix*v_ix+v_iy*v_iy+v_iz*v_iz)
        return
    end
    fPressX=Float32(0.0)
    fPressY=Float32(0.0)
    fPressZ=Float32(0.0)
    fViscX=Float32(0.0)
    fViscY=Float32(0.0)
    fViscZ=Float32(0.0)
    while stackTop>1
        s=traversalStack[i,stackTop-1]
        aBoundX=nodeABoundField[s,1]
        aBoundY=nodeABoundField[s,2]
        aBoundZ=nodeABoundField[s,3]
        bBoundX=nodeBBoundField[s,1]
        bBoundY=nodeBBoundField[s,2]
        bBoundZ=nodeBBoundField[s,3]
        clampX=min(bBoundX,max(aBoundX,x))
        clampY=min(bBoundY,max(aBoundY,y))
        clampZ=min(bBoundZ,max(aBoundZ,z))
        dx=clampX-x
        dy=clampY-y 
        dz=clampZ-z
        r2=dx*dx+dy*dy+dz*dz
        if r2<=h2
            if s<=n
                j=pointers[s]
                if j!=i && r2>0f0 && active[j]==0
                    r=sqrt(r2)
                    invr=1/r
                    #pressure
                    ux=dx*invr
                    uy=dy*invr
                    uz=dz*invr
                    rho_j=rhoField[j]
                    fPress=(p_i+pField[j])/(2*rho_j)*dW(r,h)
                    fPressX+=fPress*ux
                    fPressY+=fPress*uy
                    fPressZ+=fPress*uz
                    #viscosity
                    viscM=mu/rho_j * (d2W(r,h)+2*dW(r,h)/r)
                    fViscX+=viscM*(vel[j,1]-v_ix)
                    fViscY+=viscM*(vel[j,2]-v_iy)
                    fViscZ+=viscM*(vel[j,3]-v_iz)
                    stackTop-=1
                else
                    stackTop-=1
                end
            else
                if stackTop<=257
                    traversalStack[i,stackTop-1]=nodeChildField[s,1]
                end
                if stackTop<=256
                    traversalStack[i,stackTop]=nodeChildField[s,2]
                    stackTop+=1
                end
            end
        else
            stackTop-=1
        end
    end
    ax=(fPressX+fViscX)*mparticle/rhoField[i]
    ay=(fPressY+fViscY)*mparticle/rhoField[i]
    az=(fPressZ+fViscZ)*mparticle/rhoField[i]
    vel[i,1]+=ax*dt
    vel[i,2]+=ay*dt
    vel[i,3]+=az*dt
    vF[i]=sqrt(v_ix*v_ix+v_iy*v_iy+v_iz*v_iz)
    nothing
end

trds=min(256,n)
blks=cld(n,trds)
@cuda threads=1 bhInit(bhPos,bhVel, m, a0)
@cuda threads=trds blocks=blks posInitialize!(pos, bhPos)
@cuda threads=trds blocks=blks velInitialize!(pos,vel, m,  bhPos, bhVel)

#MAIN LOOP
function sim()
    trds=min(256,n)
    blks=cld(n,trds)
    trdsNode=min(256,nodeN)
    blksNode=cld(nodeN,trdsNode)
    #Morton coding and pointers
    mins=minimum(pos, dims=1)
    maxs=maximum(pos, dims=1)  
    @cuda threads=trds blocks=blks mortonCodeGPU!(pos,codes, mins, maxs)
    @cuda threads=trds blocks=blks resetPointers!(pointers)
    AK.sort_by_key!(codes,pointers)
    #Tree creation and accumulation
    fill!(nodeABoundField,Inf32)
    fill!(nodeBBoundField,-Inf32)
    fill!(ready,0)
    @cuda threads=trdsNode blocks=blksNode createTree!(nodeRangeField, nodeParentField, nodeChildField,
        nodeABoundField, nodeBBoundField, pos, codes, n, pointers, reversePointers)
    @cuda threads=trds blocks=blks accumulateBounds(nodeABoundField, nodeBBoundField, nodeParentField, ready)
    #SPH
    @cuda threads=trds blocks=blks getRhoP(traversalStack, k,rho0, pos,n,mparticle,h, active,rhoField,pField, pointers, nodeABoundField,nodeBBoundField,nodeChildField)
    @cuda threads=trds blocks=blks navierStokes(traversalStack, pointers, pos,vel,vF,n,dt,h,mparticle, active,rhoField,pField, mu, nodeABoundField,nodeBBoundField,nodeChildField)
    #Grav
    @cuda threads=1 postNewtonBH(bhPos, bhVel, m,dt)
    @cuda threads=trds blocks=blks gravitateParticles!(pos, bhPos, vel, n, m,  dt, active)
    nothing
end 

idx = 1:16:n
#       ^ needs to be odd, otherwise it only draws one disk

function getpoints(pos, active, idx)
    p = Array(pos[idx, :])
    a = Array(active[idx])
    pts = Point3f.(
        p[:,1],
        p[:,2],
        p[:,3]
    )
    nan = Float32(NaN)
    for j in eachindex(a)
        if a[j] == 1
            pts[j] = Point3f(nan, nan, nan)
        end
    end
    return pts
end

function getbhpoints(pos)
    p = Array(pos)
    return Point3f.(
        p[:,1],
        p[:,2],
        p[:,3]
    )
end

function getdensity(vF, idx)
    return Array(vF[idx])
end

points = Observable(getpoints(pos, active, idx))
colors = Observable(getdensity(vF, idx))

fig = Figure(size=(900,900), backgroundcolor=:black,figure_padding = 5)

ax = Axis3(fig[1,1],
    backgroundcolor=:black,
    xypanelcolor=:black,
    xzpanelcolor=:black,
    yzpanelcolor=:black,

    xlabel="X",
    ylabel="Y",
    zlabel="Z",

    xlabelcolor=:cyan,
    ylabelcolor=:red,
    zlabelcolor=:white,

    xlabelsize=22,
    ylabelsize=22,
    zlabelsize=18,

    xticklabelcolor=:white,
    yticklabelcolor=:white,
    zticklabelcolor=:gray,
    xticklabelsize=16,
    yticklabelsize=16,

    xgridcolor=(:cyan, 0.55),
    ygridcolor=(:red, 0.55),
    zgridcolor=(:white, 0.35),

    xspinecolor_1=:cyan,
    yspinecolor_1=:red,

    azimuth=0.25pi,
    elevation=0.2pi
)

limits!(ax, -a0, a0, -a0, a0, -a0, a0)

bhPoints = Observable(getbhpoints(bhPos))
scatter!(ax, points, color=colors, colormap=:plasma, colorrange=(0.0,1.0),markersize=5)
scatter!(ax, bhPoints, color=[:cyan, :red], markersize=20)
bhText = Observable("")

Label(fig[2,1], bhText,
    color=:white,
    fontsize=18,
    tellwidth=false
)
#hidedecorations!(ax)
display(fig)

statsText = Observable("Initializing...")

Label(fig[2, 1], statsText,
    color = :white,
    fontsize = 18,
    justification = :left,
    halign = :left,
    tellwidth = false
)


times= Observable(Float64[])
rActual= Observable(Float64[])
rTheory= Observable(Float64[])
rDiff= Observable(Float64[])
drs= Observable(Float64[])
hs= Observable(Float64[])

# Put the diagnostics to the right of your existing plot
# Change column 2 to whichever column is free.
diag = fig[1, 2] = GridLayout()

axR = Axis(diag[1, 1],
    title = "BH Separation",
    xlabel = "Simulation time",
    ylabel = "Separation")

axDiff = Axis(diag[2, 1],
    title = "Separation Discrepancy",
    xlabel = "Simulation time",
    ylabel = "Actual - predicted")

axDr = Axis(diag[3, 1],
    title = "Radial Velocity",
    xlabel = "Simulation time",
    ylabel = "dr/dt")

axH = Axis(diag[4, 1],
    title = "Angular Momentum",
    xlabel = "Simulation time",
    ylabel = "h")

axR.xticks = Makie.WilkinsonTicks(5)
axR.yticks = Makie.WilkinsonTicks(5)

axDr.xticks = Makie.WilkinsonTicks(5)
axDr.yticks = Makie.WilkinsonTicks(5)

axDiff.xticks = Makie.WilkinsonTicks(5)
axDiff.yticks = Makie.WilkinsonTicks(5)

axH.xticks = Makie.WilkinsonTicks(5)
axH.yticks = Makie.WilkinsonTicks(5)

axR.xticklabelcolor = :white
axR.yticklabelcolor = :white
axDr.xticklabelcolor = :white
axDr.yticklabelcolor = :white
axDiff.xticklabelcolor = :white
axDiff.yticklabelcolor = :white
axH.xticklabelcolor = :white
axH.yticklabelcolor = :white

axR.xtickcolor = :white
axR.ytickcolor = :white
axDr.xtickcolor = :white
axDr.ytickcolor = :white
axDiff.xtickcolor = :white
axDiff.ytickcolor = :white
axH.xtickcolor = :white
axH.ytickcolor = :white

axR.xlabelcolor = :white
axR.ylabelcolor = :white
axDr.xlabelcolor = :white
axDr.ylabelcolor = :white
axDiff.xlabelcolor = :white
axDiff.ylabelcolor = :white
axH.xlabelcolor = :white
axH.ylabelcolor = :white


# Plot each variable
lines!(axR, times, rActual, color = :cyan, label = "Simulation")
lines!(axR, times, rTheory, color = :orange,
       linestyle = :dash, label = "Quadrupole")
axislegend(axR)

lines!(axDiff, times, rDiff, color = :red)
hlines!(axDiff, [0.0], color = :gray, linestyle = :dash)

lines!(axDr, times, drs, color = :lime)
hlines!(axDr, [0.0], color = :gray, linestyle = :dash)

lines!(axH, times, hs, color = :yellow)

# Give the main simulation plot more space
colsize!(fig.layout, 1, Relative(0.65))
colsize!(fig.layout, 2, Relative(0.35))

rold=a0
frame=0
while true
    global rold
    global frame
    #@time CUDA.@sync begin
    frame=frame+1
    sim()
    if frame % 2 == 0
        points[] = getpoints(pos, active, idx)
        colors[] = getdensity(vF, idx)
        bhPoints[] = getbhpoints(bhPos)
        p=Array(bhPos)
        v=Array(bhVel)
        bhText[] = "BH1: x=$(round(p[1,1], digits=2)), y=$(round(p[1,2], digits=2))    " *
           "BH2: x=$(round(p[2,1], digits=2)), y=$(round(p[2,2], digits=2))"
        if frame%50==0
            
            dx=p[1,1]-p[2,1]
            dy=p[1,2]-p[2,2]
            dvx=v[1,1]-v[2,1]
            dvy=v[1,2]-v[2,2]
            r=sqrt(dx*dx+dy*dy)
            dr=(r-rold)/50
            rold=r
            dispR=a0-r
            t=frame*dt
            LOE=(a0^4-512t/5)^(1/4)
            dr = (dx*dvx + dy*dvy)/r
            h = abs(dx*dvy - dy*dvx)
            statsText[] = """
                        BH Distance:  $(round(r, digits=3))
                        Radial Disp:   $(round(dispR, digits=3))
                        Disp/Frame:   $(round(dr, digits=6))
                        LOE BH Dist: $(round(LOE, digits=3)) 
                        Dist diff:   $(round(LOE-r, digits=3))
                        h:           $(round(h, digits=3))
                        dr:        $(round(dr,digits=3))
                        Frame:        $(frame)
                        """
            push!(times[], frame)
            push!(rActual[], r)
            push!(rTheory[], LOE)
            push!(rDiff[], r-LOE)
            push!(drs[], dr)
            push!(hs[], h)
            times[]   = copy(times[])
            rActual[] = copy(rActual[])
            rTheory[] = copy(rTheory[])
            rDiff[]   = copy(rDiff[])
            drs[]   = copy(drs[])
            hs[]      = copy(hs[])
            notify(times)
            notify(rActual)
            notify(rTheory)
            notify(rDiff)
            notify(drs)
            notify(hs)
            println("Graph samples: ", length(times[]))
            println("Latest separation: ", rActual[][end])
            autolimits!(axR)
            autolimits!(axDiff)
            autolimits!(axDr)
            autolimits!(axH)
        end

    end    
    #end
end
