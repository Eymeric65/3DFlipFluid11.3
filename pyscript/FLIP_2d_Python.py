#Programme de simulation FLIP en python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import time

def Mur(A): #définition des murs

    A[0:1,:] = 1
    A[-1:,:] = 1
    A[:,0:1] = 1
    A[:,-1:] = 1

def initiateWater(n,ec,x0,y0): # donne les position et vitesse initiales des particules d'eau

    largeur = int(np.sqrt(n))

    V=np.zeros((2,largeur**2))

    x = np.linspace(0, 1, largeur) *largeur*ec +x0
    y = np.linspace(0, 1, largeur) *largeur*ec +y0
    xv, yv = np.meshgrid(x, y)

    xv= xv.flatten()

    yv = yv.flatten()

    P = np.append([xv],[yv],axis=0)

    P=P + np.random.rand(2,largeur**2)*ec/2

    return V,P

def addgrav(M,tstep): # ajout de leur gravité

    M[:,:,1] = M[:,:,1] - 9.81* tstep

def TrToGr(P,V,M,tsize,GridType,Weight): #transfert des vitesses sur la grille

    for i in range(P.shape[1]):
        Xind = int( P[0,i]/tsize)
        Yind = int( P[1,i]/tsize)

        ax = (Xind+1) - P[0,i]/tsize
        ay = (Yind+1) - P[1,i]/tsize

        GridType[Xind,Yind]=2
        GridType[Xind+1,Yind]=2
        GridType[Xind+1,Yind+1]=2
        GridType[Xind-1,Yind]=2
        GridType[Xind-1,Yind+1]=2
        GridType[Xind-1,Yind-1]=2
        GridType[Xind+1,Yind-1]=2

        M[Xind,Yind,0] = M[Xind,Yind,0] + (ax) * V[0,i]
        M[Xind,Yind,1] = M[Xind,Yind,1] + (ay) * V[1,i]

        M[Xind+1,Yind,0] = M[Xind+1,Yind,0] + (1-ax) * V[0,i]
        M[Xind,Yind+1,1] = M[Xind,Yind+1,1] + (1-ay) * V[1,i]

        Weight[Xind,Yind,0]= Weight[Xind,Yind,0]+ax
        Weight[Xind,Yind,1]= Weight[Xind,Yind,1]+ay

        Weight[Xind+1,Yind,0]= Weight[Xind+1,Yind,0]+1-ax
        Weight[Xind,Yind+1,1]= Weight[Xind,Yind+1,1]+1-ay

def normalise(M,W): #normalisation de la grille

    M = np.where(W!=0,M/W,M)

    return M

def TrToPr(P,V,M,OldM,tsize,flipness): #transfert des vitesses de la grille dans les particules

    for i in range(P.shape[1]):

        Xind = int( P[0,i]/tsize)
        Yind = int( P[1,i]/tsize)

        ax = (Xind+1) - P[0,i]/tsize
        ay = (Yind+1) - P[1,i]/tsize

        xvit = M[Xind,Yind,0] * (ax) + M[Xind+1,Yind,0] * (1-ax)
        yvit = M[Xind,Yind,1] * (ay) + M[Xind,Yind+1,1] * (1-ay)

        oxvit = OldM[Xind,Yind,0] * (ax) + OldM[Xind+1,Yind,0] * (1-ax)
        oyvit = OldM[Xind,Yind,1] * (ay) + OldM[Xind,Yind+1,1] * (1-ay)

        V[0,i] = xvit * (1-flipness) + flipness* (V[0,i] + xvit-oxvit)
        V[1,i] = yvit * (1-flipness) + flipness* (V[1,i] + yvit-oyvit)

def integrate(P,V,tstep): #intégration d'euler (propre à la maquette en python)

    for i in range(P.shape[1]):

        P[0,i] = P[0,i] + tstep * V[0,i]
        P[1,i] = P[1,i] + tstep * V[1,i]

def Boundaries(TypeGrid,VStag): # conditions au limites

    VStag[:-1,:-1,0] = np.where(TypeGrid==1,0,VStag[:-1,:-1,0])
    VStag[1:,:-1,0] = np.where(TypeGrid==1,0,VStag[1:,:-1,0])

    VStag[:-1,:-1,1] = np.where(TypeGrid==1,0,VStag[:-1,:-1,1])
    VStag[:-1,1:,1] = np.where(TypeGrid==1,0,VStag[:-1,1:,1])

def divcal(M,tsize,Weight,dens,tstep,rho): # calcul de la divergence

    density = (Weight[1:,:-1,0] + Weight[:-1,:-1,0] + Weight[:-1,1:,1] + Weight[:-1,:-1,1] )/2

    DivGrid = (( M[1:,:-1,0] - M[:-1,:-1,0] + M[:-1,1:,1] - M[:-1,:-1,1] ) - np.maximum(density-dens,np.zeros(density.shape)) )*rho*tsize/tstep

    return DivGrid

def Jaciter(PB,Div,WallC,type): # étape de jacobi

    PA = np.where( (WallC<3 ),(np.roll(PB,1,axis=1)+np.roll(PB,-1,axis=1)+np.roll(PB,1,axis=0)+np.roll(PB,-1,axis=0) - Div )/(4-WallC),PB)

    return PA

def compareN(Div,PB,WallC,type): #fonction qui retourne l'erreur de la solution trouvé par rapport à l'idéal

    return (np.linalg.norm( np.where((WallC<3 ),Div- ( - (4-WallC) * PB + np.roll(PB,1,axis=1)+np.roll(PB,-1,axis=1)+np.roll(PB,1,axis=0)+np.roll(PB,-1,axis=0) ),0)/np.linalg.norm(PB)))


def addpress(M,P,tsize,tstep,rho): #ajout de la pression

    M[1:-1,:-1,0] = M[1:-1,:-1,0] -  (P[1:,:]-P[:-1,:] )*tstep/(tsize*rho)

    M[:-1,1:-1,1] = M[:-1,1:-1,1] - (P[:,1:]-P[:,:-1] )*tstep/(tsize*rho)

def divergence(M,tsize): # divergence discrète

    return np.linalg.norm((np.roll(M[:,:,0],-1,axis=0)-M[:,:,0]+np.roll(M[:,:,0],-1,axis=1)-M[:,:,1] )/tsize)

#---------affichage---------

fig=plt.figure()

Situation = fig.add_subplot(111)

plt.ion()

vecscale=2

def dotsize(n) :

    return (n*taillepointnormal/2)**2

#---------------------------------

#-----------paramètre de simulation---------

Npart = 3000

NJac= 70

TailleSim = (80,50)

tsize=10

dens=4

rho=4

tstep=0.08

flipness=0.85

#------------------------------------------

PartV,PartP=initiateWater(Npart,tsize/1.5,tsize*30.2,tsize*5.2)

boundaries = [0,TailleSim[0]*tsize , 0,TailleSim[0]*tsize ]

windowsize = [9.5,9.5]

nbpixel = windowsize[0]*fig.dpi

taillepointnormal = nbpixel/ (-boundaries[0]+boundaries[1]) * 1


t=True

while True :

    t=False

    ExecTime= time.time()

    Situation.cla()

    #initialisation des matrices
    TypeGrid = np.zeros(TailleSim)

    PAgrid = np.zeros(TailleSim)

    StagGrid = np.zeros((TailleSim[0]+1,TailleSim[1]+1,2))
    StagWeight = np.zeros((TailleSim[0]+1,TailleSim[1]+1,2))
    DivGrid = np.ones(TailleSim)

    #-----------


    Situation.axis(boundaries)


    #1 on place les particules sur la grilles et 2 on définit les cases de fluide
    TrToGr(PartP,PartV,StagGrid,tsize,TypeGrid,StagWeight)


    StagGrid = normalise(StagGrid,StagWeight)

    OldVit = np.copy(StagGrid)
    #------------

    Mur(TypeGrid) #2 On définit les murs de notre grille

    OnlyWall = np.where(TypeGrid==1,1,0)

    WallC = np.roll(OnlyWall,1,axis=0)+np.roll(OnlyWall,-1,axis=0)+np.roll(OnlyWall,1,axis=1)+np.roll(OnlyWall,-1,axis=1)


    addgrav(StagGrid,tstep) # 3.a ajout de la gravité

    Boundaries(TypeGrid,StagGrid) #3.b les CL

    #3.c calcul de la pression
    DivGrid = divcal(StagGrid,tsize,StagWeight,dens,tstep,rho)

    for i in range(NJac):

        PAgrid = Jaciter(PAgrid,DivGrid,WallC,TypeGrid)

    #---------

    #3.d ajout de la pression
    addpress(StagGrid,PAgrid,tsize,tstep,rho)

    Boundaries(TypeGrid,StagGrid) #3.b les CL

    #4 intégration

    TrToPr(PartP,PartV,StagGrid,OldVit,tsize,flipness)
    integrate(PartP,PartV,tstep)

    #sortir les indices
    Water = np.nonzero(TypeGrid==2)
    Walls = np.nonzero(TypeGrid==1) #sortir les indices de mur

    Pressure = np.nonzero(PAgrid!=0)

    #print(PAgrid)

    #-----------------

    #print(StagGrid[Water[0]+1,Water[1]])

    Situation.scatter(Walls[0]*tsize+tsize/2,Walls[1]*tsize+tsize/2,marker="s",s=dotsize(tsize),c="grey") #afficher les murs

    Situation.scatter(Water[0]*tsize+tsize/2,Water[1]*tsize+tsize/2,marker="s",s=dotsize(tsize),c="aqua",alpha=0.2) #afficher l'eau

    Situation.scatter(Pressure[0]*tsize+tsize/2,Pressure[1]*tsize+tsize/2,s=PAgrid[Pressure]*10**-3,c="red",alpha=0.1)

    Situation.scatter(PartP[0,:],PartP[1,:],c="b",alpha=0.5,s=0.5) # afficher les particules

    fig.set_size_inches(windowsize)

    plt.show()

    print("%.2e" % compareN(DivGrid,PAgrid,WallC,TypeGrid),"divergence calc","%.3f" % (time.time()-ExecTime),"exec time","%.3f" % np.linalg.norm(divcal(StagGrid,tsize,StagWeight,dens,tstep,rho))  )

    plt.pause(0.001)