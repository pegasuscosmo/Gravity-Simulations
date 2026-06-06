import taichi as ti
import tkinter as tk
ti.reset()
ti.init(arch=ti.cpu,default_fp=ti.f32)
#tk init
root=tk.Tk()
root.title("Generation Editor")
root.geometry("300x500")

#physics
G=1 #6.6743e-20 #irl, in km
dt=1
e=10 #softening for majors
#   trajectories
steps=1000
dtT=dt #different dtT causes 'wiggly' trajectories when close proximity involved

#ti window
sW,sH=1620,950 #screen resolution
window = ti.ui.Window("Ring Semi-NBody Sim", (sW, sH))
canvas = window.get_canvas()
gui = window.get_gui()

#generation
particleN=1000000
majorN=100 #maximum

#default
innerRad=100
outerRad=500

rings=[
    #inner  outer   % of particles
    [100,   500,    50],
    [200,   400,    50]
]

bodies=[ #default
    #start pos      start vel   mass    radius
    [sW/2,sH/2,      0,0,      100,     0.01],    #parent
]

#colors
particleColor=ti.Vector([0.4,0.4,0.4])
brightCutoff=0.3 #brightness cutoff for major bodies

#fields
#   particles
posField=ti.Vector.field(2,dtype=ti.f32,shape=particleN) #particle positions
velField=ti.Vector.field(2,dtype=ti.f32,shape=particleN) #particle velocities
enabledField=ti.field(dtype=ti.u8,shape=particleN) #enabled particles - disabled on collision
#   majors
majorNCurrent=ti.field(dtype=ti.u64,shape=()) #taichi var
majorNCurrent[None]=len(bodies) #number of bodies
majorPosField=ti.Vector.field(2,dtype=ti.f32,shape=majorN) #major positions
majorVelField=ti.Vector.field(2,dtype=ti.f32,shape=majorN) #major velocities
majorMassField=ti.field(dtype=ti.f32,shape=majorN) #major masses
majorRadii=ti.field(dtype=ti.f32,shape=majorN) #major radii
majorColors=ti.Vector.field(3,dtype=ti.f32,shape=majorN) #major colors
#   rings
ringVals=ti.Vector.field(3,dtype=ti.f32,shape=10)
ringN=ti.field(dtype=ti.int8,shape=())
#   trajectories
trajectories=ti.Vector.field(2,dtype=ti.f32,shape=(steps,majorN)) #major x steps trajectory field
trajPos=ti.Vector.field(2,dtype=ti.f32,shape=majorN) #projected positions in trajectories 
trajPosBckp=ti.Vector.field(2,dtype=ti.f32,shape=majorN) #backup projected positions for force calculation
trajVel=ti.Vector.field(2,dtype=ti.f32,shape=majorN) #projected velocities
vertices=ti.Vector.field(2,dtype=ti.f32,shape=(steps*majorN)) #major*steps trajectory field, for drawing
trajColors=ti.Vector.field(3,dtype=ti.f32,shape=(steps*majorN)) #per vertex color field
#   body creation
posN=ti.Vector([0.0,0.0]) #position
drawPosN=ti.Vector.field(2,dtype=ti.f32,shape=1) #draw position
velN=ti.Vector([0.0,0.0]) #velocity
massN=0 #mass
radN=0 #radius
colorN=(0,0,0) #color
trajN=ti.Vector.field(2,dtype=ti.f32,shape=steps) #projected position

def listToText(l):
    s=""
    for i in l:
        s+=str(i)+"\n"
    return s

@ti.kernel
def initColors():
    majorColors[0]=ti.Vector([1,1,1]) #first body white
    for i in range(majorN-1):
        j=i+1
        bright=-1
        r=1.0
        g=1.0
        b=1.0
        while bright<brightCutoff: #avoid dark colors
            r=ti.random()
            g=ti.random()
            b=ti.random()
            bright=ti.sqrt(r*r+g*g+b*b)

        majorColors[j]=ti.Vector([r,g,b]) #set random color
initColors() #only called once
majorColors[0]=ti.Vector([1,1,1])

@ti.kernel
def initTI():
    COM=ti.Vector([0.0,0.0]) #Center of major mass vector
    m=0 #total mass
    for i in range(majorNCurrent[None]):
        COM+=majorPosField[i]*majorMassField[i]
        m+=majorMassField[i]
    COM/=m #COM formula
    n=0
    ti.loop_config(serialize=True)
    for ri in range(ringN[None]):
        ring=ringVals[ri]
        np=ti.cast(particleN*ring[2],ti.int32)
        ti.loop_config(serialize=False)
        for i in range(np):
            innerRad=ring[0]
            outerRad=ring[1]
            r=0
            x=0
            y=0
            d=ti.Vector([0,0])
            while r<innerRad or r>outerRad: #stay inside ring
                x=(ti.random()-0.5)*outerRad*10+sW/2
                y=(ti.random()-0.5)*outerRad*10+sH/2
                d=COM-ti.Vector([x,y]) #use COM as center point
                r=d.norm()
            posField[n+i]=ti.Vector([x,y])
            s=ti.sqrt(G*m/r) #orbital speed for circular orbit
            velField[n+i]=ti.Vector([d[1], -d[0]])/r*s #set velocity perpendicular to d
        n+=np
    for s in range(steps):
        for i in range(majorNCurrent[None]):
            trajColors[s*majorN+i]=majorColors[i] #set trajectory colors


#tkinter UI and functions
def submit(idx=-1):
    pos=posEntry.get()
    vel=velEntry.get()
    mass=massEntry.get()
    rad=radiusEntry.get()
    if pos=="":
        errorLabel.config(text="No Position Given")
        return
    if vel=="":
        errorLabel.config(text="No Velocity Given")
        return
    if mass=="":
        errorLabel.config(text="No Mass Given")
        return
    if rad=="":
        errorLabel.config(text="No Radius Given")
        return
    if not mass.isnumeric():
        errorLabel.config(text=f"Invalid Mass {mass}")
    if not rad.isnumeric():
        errorLabel.config(text=f"Invalid Radius {rad}")
    t=""
    for i in pos:
        if i==",":
            x=t
            t=""
        elif i!=" ":
            t+=i
    y=t
    t=""
    for i in vel:
        if i==",":
            xv=t
            t=""
        elif i!=" ":
            t+=i
    yv=t
    if not x.isnumeric():
        errorLabel.config(text=f"Invalid X Coordinate {x}")
    if not y.isnumeric():
        errorLabel.config(text=f"Invalid Y Coordinate {y}")
    if not xv.isnumeric():
        errorLabel.config(text=f"Invalid X Velocity {xv}")
    if not yv.isnumeric():
        errorLabel.config(text=f"Invalid Y Velocity {yv}")
    x=float(x)
    y=float(y)
    xv=float(xv)
    yv=float(yv)
    mass=float(mass)
    rad=float(rad)
    l=[x,y,xv,yv,mass,rad]
    errorLabel.config(text=" ")
    if idx!=-1:
        bodies[idx]=l
    else:
        bodies.append(l)
    bodyLabel.config(text=f"Bodies: {listToText(bodies)}")

def remove():
    idx=indexEntry.get()
    indexEntry.delete(0,len(idx))
    if idx!="" and (idx.capitalize()!=idx or idx!=idx.lower() or int(idx)>=len(bodies)):
        errorLabel.config(text=f"Invalid Index: {idx}")
    else:
        if idx=="":
            idx=-1
        else:
            idx=int(idx)
        if len(bodies)>0:
            bodies.pop(idx)
            bodyLabel.config(text=f"Bodies: {listToText(bodies)}")
            errorLabel.config(text=" ")
        else:
            errorLabel.config(text="Nothing to Remove")

def replace():
    idx=indexEntry.get()
    if idx!="" and (idx.capitalize()!=idx or idx!=idx.lower() or int(idx)>=len(bodies)):
        errorLabel.config(text=f"Invalid Index: {idx}")
    elif len(bodies)>0:
        if idx=="":
            remove()
            submit()
        else:
            submit(int(idx))
        errorLabel.config(text=" ")
    else:
        errorLabel.config(text="Nothing to Replace")

def submitRing(idx=-1):
    inn=innerEntry.get()
    out=outerEntry.get()
    per=percentEntry.get()
    if inn=="" or out=="" or per=="":
        errorLabel.config(text="Nothing to Append")
    l=[float(inn),float(out),float(per)]
    errorLabel.config(text=" ")
    if idx!=-1:
        rings[idx]=l
    else:
        rings.append(l)
    ringLabel.config(text=f"Rings: {listToText(rings)}")

def removeRing():
    idx=indexRingEntry.get()
    indexRingEntry.delete(0,len(idx))
    if idx!="" and (idx.capitalize()!=idx or idx!=idx.lower() or int(idx)>=len(bodies)):
        errorLabel.config(text=f"Invalid Index: {idx}")
        return
    if idx=="":
        idx=-1
    else:
        idx=int(idx)
    if len(rings)>0:
        rings.pop(idx)
        ringLabel.config(text=f"Rings: {listToText(rings)}")
        errorLabel.config(text=" ")
    else:
        errorLabel.config(text="Nothing to Remove")

def replaceRing():
    idx=indexRingEntry.get()
    if idx!="" and (idx.capitalize()!=idx or idx!=idx.lower() or int(idx)>=len(rings)):
        errorLabel.config(text=f"Invalid Index: {idx}")
    elif len(rings)>0:
        if idx=="":
            removeRing()
            submitRing()
        else:
            submitRing(int(idx))
        errorLabel.config(text=" ")
    else:
        errorLabel.config(text="Nothing to Replace")

#Tkinter UI
#   Body Creation
#       Labels
posLabel=tk.Label(root,text="Position x,y")
posLabel.grid(row=0,column=0)
velLabel=tk.Label(root,text="Velocity xv,yv")
velLabel.grid(row=1,column=0)
massLabel=tk.Label(root,text="Mass")
massLabel.grid(row=2,column=0)
radiusLabel=tk.Label(root,text="Radius")
radiusLabel.grid(row=3,column=0)
#       Entries
posEntry=tk.Entry(root)
posEntry.grid(row=0,column=1,columnspan=3)
velEntry=tk.Entry(root)
velEntry.grid(row=1,column=1,columnspan=3)
massEntry=tk.Entry(root)
massEntry.grid(row=2,column=1,columnspan=3)
radiusEntry=tk.Entry(root)
radiusEntry.grid(row=3,column=1,columnspan=3)
#       Index
indexEntry=tk.Entry(root)
indexEntry.grid(row=4,column=1,columnspan=3)
indexLabel=tk.Label(root,text="Remove/Replace Index")
indexLabel.grid(row=4,column=0,columnspan=1)
#       Add/Remove/Replace Buttons
addButton=tk.Button(root,command=submit,text="Append")
addButton.grid(row=5,column=1)
removeButton=tk.Button(root,command=remove,text="Remove")
removeButton.grid(row=5,column=2)
replaceButton=tk.Button(root,command=replace,text="Replace")
replaceButton.grid(row=5,column=3)
#       List display
bodyLabel=tk.Label(root,text=f"Bodies: {listToText(bodies)}")
bodyLabel.grid(row=6,column=0,columnspan=4)
#   Rings
#       Labels
innerLabel=tk.Label(root,text="Inner Ring Radius")
innerLabel.grid(row=7,column=0)
outerLabel=tk.Label(root,text="Outer Ring Radius")
outerLabel.grid(row=8,column=0)
percentLabel=tk.Label(root,text="Percent of Particles")
percentLabel.grid(row=9,column=0)
indexRingLabel=tk.Label(root,text="Remove/Replace Index")
indexRingLabel.grid(row=10,column=0)
#       Entries
innerEntry=tk.Entry(root)
innerEntry.grid(row=7,column=1,columnspan=3)
outerEntry=tk.Entry(root)
outerEntry.grid(row=8,column=1,columnspan=3)
percentEntry=tk.Entry(root)
percentEntry.grid(row=9,column=1,columnspan=3)
indexRingEntry=tk.Entry(root)
indexRingEntry.grid(row=10,column=1,columnspan=3)
#       Add/Remove/Replace Buttons
ringSubmitButton=tk.Button(root,text="Append",command=submitRing)
ringSubmitButton.grid(row=11,column=1)
ringRemoveButton=tk.Button(root,text="Remove",command=removeRing)
ringRemoveButton.grid(row=11,column=2)
ringReplaceButton=tk.Button(root,text="Replace",command=replaceRing)
ringReplaceButton.grid(row=11,column=3)
#       List display
ringLabel=tk.Label(root,text=f"Rings: {listToText(rings)}")
ringLabel.grid(row=12,column=0,columnspan=4)
#   Error display
errorLabel=tk.Label(root,text=" ")
errorLabel.grid(row=13,column=0,columnspan=4)

def init():
    global posN,velN,massN,radN,colorN
    global body,LMB,paused #globals to avoid local variable assignment
    global innerRad, outerRad

    iRtemp=innerEntry.get() #outputs string
    if iRtemp!="": #if nothing in the text entry, use default
        innerRad=float(iRtemp)
    oRtemp=outerEntry.get() #same as inner radius
    if oRtemp!="":
        outerRad=float(oRtemp)

    if bodies==[] or rings==[]: #do not generate on empty major or rings list
        errorLabel.config(text="Nothing to Generate")
        return
    if len(rings)>10: #do not generate on overfilled rings list
        errorLabel.config(text="Too Many Rings")
        return
    totalPercent=0
    ringN[None]=len(rings)
    for i in range(len(rings)): #add ring values to taichi field
        r=rings[i]
        totalPercent+=r[2]
        if r[2]<0: #do not generate on negative percents
            errorLabel.config(text="Cannot Generate With Negative Percents")
            return
        if totalPercent>100: #do not generate on >100% rings
            errorLabel.config(text="Cannot Generate With >100%")
            return
        if r[0]>r[1]:
            errorLabel.config(text=f"Ring {i}'s Inner Radius is Larger than Outer Radius")
            return
        if r[0]+10>r[1]:
            errorLabel.config(text=f"Ring {i} Too Thin")
            return
        ringVals[i]=ti.Vector([r[0],r[1],r[2]/100])
    
    errorLabel.config(text="") #reset error text in tkinter if works
    #reset vars and fields
    pixels.fill(0); drawMajorPos.fill(0); drawMajorRadii.fill(0)
    body=False; LMB=False; paused=False
    drawPosN[0]=ti.Vector([0.0,0.0])
    posN=ti.Vector([0.0,0.0]); velN=ti.Vector([0.0,0.0])
    massN=0; radN=0; colorN=(0,0,0)
    enabledField.fill(1); majorPosField.fill(0)
    majorVelField.fill(0); majorMassField.fill(0)
    majorRadii.fill(0); trajectories.fill(0)
    vertices.fill(0); trajN.fill(0)
    majorNCurrent[None]=len(bodies)

    # init for majors
    for i in range(majorNCurrent[None]):
        b=bodies[i]
        majorPosField[i]=ti.Vector([b[0],b[1]])
        majorVelField[i]=ti.Vector([b[2],b[3]])
        majorMassField[i]=b[4]
        majorRadii[i]=b[5]
    initTI()
#Generation button
updSimButton=tk.Button(root,text="Generate",command=init)
updSimButton.grid(row=14,column=0,columnspan=4)

@ti.kernel
def stepParticles():
    for i in range(particleN):
        if enabledField[i]==1:
            posField[i]+=velField[i]*dt/2 #verlet pos half step
            a=ti.Vector([0.0,0.0])
            pos=posField[i]
            for j in range(majorNCurrent[None]): #accumulate forces
                m=majorMassField[j]
                tpos=majorPosField[j]
                d=tpos-pos
                r=d.norm()
                if r<majorRadii[j]*sH:
                    enabledField[i]=ti.u8(0)
                a+=m*G*d/(r*r*r)
            velField[i]+=a*dt #velocity verlet full step
            posField[i]+=velField[i]*dt/2 #verlet pos half step

@ti.kernel
def stepMajors():
    for i in range(majorNCurrent[None]):
        majorPosField[i]+=majorVelField[i]*dt/2 #verlet pos half step 
        #done independent of main force loop to avoid race conditions
    for i in range(majorNCurrent[None]):
        pos=majorPosField[i]
        a=ti.Vector([0.0,0.0])
        for j in range(majorNCurrent[None]): #force accumulate
            if i!=j:
                tpos=majorPosField[j]
                m=majorMassField[j]
                d=tpos-pos
                r=d.norm()+e
                a+=m*G*d/(r*r*r)
        majorVelField[i]+=a*dt
    for i in range(majorNCurrent[None]): #verlet pos half step
        majorPosField[i]+=majorVelField[i]*dt/2 #same as first one

pixels=ti.Vector.field(3,dtype=ti.f32,shape=(sW,sH))
drawMajorPos=ti.Vector.field(2,dtype=ti.f32,shape=majorN)
drawMajorRadii=ti.field(dtype=ti.f32,shape=majorN)
@ti.kernel
def renderImg(zoom: ti.f32, off: ti.types.vector(2,dtype=ti.f32)):
    center=ti.Vector([sW/2,sH/2])
    cam=ti.Vector([off[0]*sW,off[1]*sH])
    for i in range(particleN):
        if enabledField[i]==1:
            pos=posField[i]
            drawPos=(pos-center+cam)*zoom+center #convert world space to screen space
            x=ti.cast(drawPos[0],ti.int64)
            y=ti.cast(drawPos[1],ti.int64)
            if x>0 and y>0 and x<sW and y<sH:
                pixels[x,y]=particleColor #set pixel color
    for i in range(majorNCurrent[None]):
        p=(majorPosField[i]-center+cam)*zoom+center #convert world space to screen space
        drawMajorPos[i]=ti.Vector([p[0]/sW,p[1]/sH]) #set vector in field
        drawMajorRadii[i]=majorRadii[i]*zoom #sale radius by zoom

@ti.kernel
def plotTrajectories(zoom: ti.f32, off: ti.types.vector(2,dtype=ti.f32)):
    for i in range(majorNCurrent[None]): #set temp vectors
        trajPos[i]=majorPosField[i]
        trajVel[i]=majorVelField[i]
    ti.loop_config(serialize=True) #serialize main loop
    for s in range(steps):
        ti.loop_config(serialize=False)
        for i in range(majorNCurrent[None]): #verlet pos half step
            trajPos[i]+=trajVel[i]*dtT/2
            trajPosBckp[i]=trajPos[i] #copy positions to backup field
        ti.loop_config(serialize=False)
        for i in range(majorNCurrent[None]):
            pos=trajPosBckp[i]
            a=ti.Vector([0.0,0.0])
            for j in range(majorNCurrent[None]): #force accumulate
                if i!=j:
                    tpos=trajPos[j]
                    d=tpos-pos
                    r=d.norm()+e
                    m=majorMassField[j]
                    a+=m*G*d/(r*r*r)
            trajVel[i]+=a*dtT
            trajPosBckp[i]+=trajVel[i]*dtT/2 #only update backup to avoid race conditions
            trajectories[s,i]=trajPosBckp[i] #add to trajectories
        ti.loop_config(serialize=False)
        for i in range(majorNCurrent[None]):
            trajPos[i]=trajPosBckp[i] #copy backup positions to projected position
    center=ti.Vector([sW/2,sH/2])
    cam=ti.Vector([off[0]*sW,off[1]*sH])
    for s in range(steps):
        ti.loop_config(serialize=False)
        for i in range(majorNCurrent[None]):
            p=(trajectories[s,i]-center+cam)*zoom+center #convert to screen space
            p=ti.Vector([p[0]/sW,p[1]/sH])
            vertices[s*majorN+i]=p

@ti.kernel
def addBodyN(pos: ti.types.vector(2,dtype=ti.f32), vel: ti.types.vector(2,dtype=ti.f32), 
             mass: ti.f32, rad: ti.f32, color: ti.types.vector(3,dtype=ti.f32), 
             off: ti.types.vector(2,dtype=ti.f32), zoom: ti.f32):
    q=majorNCurrent[None]
    center=ti.Vector([sW/2,sH/2])
    cam=ti.Vector([off[0]*sW,off[1]*sH])
    pixelPos = ti.Vector([pos[0]*sW, pos[1]*sH]) #convert to screen space
    p=(pixelPos-center)/zoom +center-cam
    majorPosField[q]=p #add values to major fields
    majorVelField[q]=vel
    majorMassField[q]=mass
    majorRadii[q]=rad
    majorColors[q]=color
    for s in range(steps): 
        trajColors[s*majorN+q]=color
    majorNCurrent[None]+=1

@ti.kernel
def newBodyTrajectory(pos: ti.types.vector(2,dtype=ti.f32), vel: ti.types.vector(2,dtype=ti.f32), off: ti.types.vector(2,dtype=ti.f32), zoom: ti.f32):
    center=ti.Vector([sW/2,sH/2])
    cam=ti.Vector([off[0]*sW,off[1]*sH])
    pixelPos = ti.Vector([pos[0]*sW, pos[1]*sH]) #convert to world space
    p=(pixelPos-center)/zoom +center-cam
    posTemp=p
    velTemp=vel
    ti.loop_config(serialize=True) #same loop as major trajectories
    for s in range(steps):
        posTemp+=velTemp*dtT/2 #position doesnt need to be updated in seperate loop, major body trajectories not affected
        a=ti.Vector([0.0,0.0])
        for i in range(majorNCurrent[None]):
            m=majorMassField[i]
            tpos=trajectories[s,i]
            d=tpos-posTemp
            r=d.norm()+e
            a+=d*m*G/(r*r*r)
        velTemp+=a*dtT
        posTemp+=velTemp*dtT/2
        trajN[s]=posTemp
    for s in range(steps):
        p=(trajN[s]-center+cam)*zoom+center #convert back to screen space for drawing
        p=ti.Vector([p[0]/sW,p[1]/sH])
        trajN[s]=p
        
#camera and control variables
zoom=0.95
moveS=0.01
offset=ti.Vector([0.0,0.0])
#booleans for body creation and pausing
body=False
LMB=False
paused=False
esc=False

sqrt2=1.414
init()

#Sensitivity scales and labels
camScale=tk.Scale(root,from_=25,to=400,orient='horizontal')
camScale.grid(row=15,column=1,columnspan=3)
camLabel=tk.Label(root,text="Camera Sensitivity %")
camLabel.grid(row=15,column=0)
createScale=tk.Scale(root,from_=10,to=500,orient='horizontal')
createScale.grid(row=16,column=1,columnspan=3)
createLabel=tk.Label(root,text="Creation Sensitivity %")
createLabel.grid(row=16,column=0)
camScale.set(100)
createScale.set(100)

# ----------------------------------------------------- MAIN LOOP -----------------------------------------------------

while window.running:
    root.update_idletasks()
    root.update()

    camSense=camScale.get()
    createSense=createScale.get() #*1e20 #for irl masses

    window.get_events()
    #ui, pausing and creation
    with gui.sub_window("Toggles", 0.,0, 0.1, 0.1):
        if not paused:
            if gui.button("Pause"):
                esc=True
                paused=True
        else:
            if gui.button("Unpause"):
                paused=False
                esc=True
        if body:
            if gui.button("Disable Creation"):
                body=False
                esc=True
        else:
            if gui.button("Enable Creation"):
                body=True
                esc=True

    #camera movement
    if window.is_pressed('k'):
        zoom /= 1.05**(camSense/100)
    if window.is_pressed('i'):
        zoom *= 1.05**(camSense/100)
    if window.is_pressed('w'):
        offset[1]-=camSense/100*moveS/zoom/sH*1000
    if window.is_pressed('s'):
        offset[1]+=camSense/100*moveS/zoom/sH*1000
    if window.is_pressed('a'):
        offset[0]+=camSense/100*moveS/zoom/sW*1000
    if window.is_pressed('d'):
        offset[0]-=camSense/100*moveS/zoom/sW*1000    

    if majorNCurrent[None]>=majorN and body:
        body=False #disable if major cap reached
    offset.normalized() #set speed of camera to 1

    if not paused:
        #physics step if not paused
        stepParticles()
        stepMajors()

    pixels.fill(0) #clear render
    renderImg(zoom,offset)
    plotTrajectories(zoom, offset)

    #body creation logic
    if window.is_pressed(ti.ui.LMB) and (body) and (not esc):
        cursor=ti.Vector(window.get_cursor_pos())
        if not LMB: #on the first frame of the click
            LMB=True
            colorN=tuple(majorColors[majorNCurrent[None]]) #convert color to tuple
            posN=cursor #set position
        else:
            velN=posN-cursor #set velocity
            velN*=10
            #creation controls
            if window.is_pressed('t'):
                massN+=createSense/100*10
            if window.is_pressed('g'):
                massN-=createSense/100*10
                massN=max(0,massN)
            if window.is_pressed('h'):
                radN+=createSense/100/1000
            if window.is_pressed('f'):
                radN-=createSense/100/1000
                radN=max(0,radN)
            if window.is_pressed(ti.ui.ESCAPE):
                esc=True #cancel
            newBodyTrajectory(posN,velN,offset,zoom) #plot trajectory
        drawPosN[0]=posN #set draw position
    elif LMB:
        LMB=False #reset flag
        if not esc: #if not cancelled add body
            addBodyN(posN,velN,massN,radN,ti.Vector(colorN),offset,zoom)
            #reset variables
            posN=ti.Vector([0.0,0.0])
            drawPosN[0]=posN
            velN=ti.Vector([0.0,0.0])
            massN=0
            radN=0
            colorN=ti.Vector([0,0,0])
            trajN.fill(0)
    if not window.is_pressed(ti.ui.LMB):
        esc=False #reset flag

    #draw
    canvas.set_image(pixels)
    canvas.circles(vertices,radius=0.0005,per_vertex_color=trajColors)
    canvas.circles(drawMajorPos,1,per_vertex_radius=drawMajorRadii,per_vertex_color=majorColors)
    if LMB and colorN!=(0,0,0): #draw created body
        canvas.circles(drawPosN,radius=radN*zoom,color=colorN)
        canvas.circles(trajN,radius=0.0005,color=colorN)
    
    #info boxes
    with gui.sub_window("Creation Menu", 0.2,0, 0.1, 0.1):
        gui.text(f"Mass: {massN:.1f}")
        gui.text(f"Radius: {radN:.4f}")
        gui.text(f"Velocity: {velN.norm():.2f}")
    with gui.sub_window("Controls: ", 0.1,0, 0.1, 0.1):
        gui.text("Cam movement: WASD\nZoom: IK\nCreation: hold LMB\n- Mass: TG\n- Radius: HF")

    #print(window.get_window_shape()) #use to find window size
    #update display
    window.show()
