#Ce code permet d'exporter les indices des cases qui vont être durant la simulation des solides


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os

exportfile = "D:/bureau/prepa/TIPE/Collide setter/IndicesCampanaFin2.txt" # lien du fichier a créer

#---------------Affichage ---------
fig = plt.figure()
ax1 = fig.gca(projection='3d')

def dotsize(n) :

    return (n*taillepointnormal/2)**2

#----------------------------------

#Fonction qui retourne True/False selon si le point c et proche a une distance r du triangle p1,p2,p3
def traverse3D(pr1,pr2,pr3,c,r):

    ex1 = pr2-pr1
    ex2=pr2-pr3
    ex3=pr3-pr1

    ex1 = ex1/np.linalg.norm(ex1)
    ex2 = ex2/np.linalg.norm(ex2)
    ex3 = ex3/np.linalg.norm(ex3)

    p1 = pr1 -ex1*r -ex3*r

    p2 = pr2 + ex1*r +ex2*r

    p3 = pr3 + ex3*r - ex2*r

    n1 = np.cross(p3-p1,p2-p1)
    n1 = n1/np.linalg.norm(n1)

    n2 = np.cross(p2-p3,p1-p3)

    d=np.sum(n1*(c-p1))

    v1=np.cross(p3-p1,c-p1)

    v2 = - np.cross(p2-p1,c-p1)

    v3=np.cross(p2-p3,c-p3)

    pr1 = np.sum(n1*v1)

    pr2 = np.sum(n1*v2)

    pr3 = np.sum(n2*v3)

    return (pr1>=0) &(pr2>=0)&(pr3>=0)&(abs(d)<=r)

#Fonction qui applique traverse3D a la matrice des Indices (fonction optimisé qui n'applique traverse3D que dans les cases proches au triangle)
def vtraverse3D(p1,p2,p3,Ind,tsize,r):

    tm = (p1+p2+p3)/3

    mr = int(( r + max(np.linalg.norm(p1-tm),np.linalg.norm(p2-tm),np.linalg.norm(p3-tm))) /tsize)+1

    cx = int(tm[0]/tsize)

    cy = int(tm[1]/tsize)

    cz = int(tm[2]/tsize)

    res = np.zeros(Ind.shape[1:])

    for i in range(cx - mr,cx+mr+1):

        for j in range(cy - mr,cy+mr+1):

            for l in range(cz - mr,cz+mr+1):

                if((i>=0)&(j>=0)&(l>=0)&(l<Ind.shape[3])&(j<Ind.shape[2])&(i<Ind.shape[1])):

                    res[i,j,l] = traverse3D(p1,p2,p3,(Ind[:,i,j,l]+np.array([0.5,0.5,0.5]))*tsize,r)

    return res

#paramètre de la simulation
TailleSim = (200,200,80)

tsize = 1
#-------------------------

Solid = np.zeros(TailleSim)
Indices = np.indices(TailleSim)

file = open("D:/bureau/prepa/TIPE/Collide setter/lac de campanadeuxieme essaie.stl",'r')

triangle = []

line = file.readline() #lecture du fichier .stl

Tr=0

points = []

while line!="": #boucle qui extrait les triangle du fichier .stl

    if line == "endloop\n":

        Tr=0

        triangle+=[points]

        points = []

    if Tr == 1:

        points += [np.array(list(map( float ,line.split(" ")[1:])))]


    if line == "outer loop\n":

        Tr=1

    line = file.readline()

print("fin text ",len(triangle))
file.close()

c=0

for triang in triangle: #appliquer vtraverse3D a tout les triangles
    c+=1
    if(c/100==int(c/100)):
        print(c)

    Solid+= vtraverse3D(triang[0],triang[1],triang[2],Indices,tsize,tsize/np.sqrt(2))

Walls = np.nonzero(Solid)

exp = open(exportfile,"w")


for i in range(len(Walls[0])): #écriture du fichier

    exp.write(str(Walls[0][i])+" "+str(Walls[1][i])+" "+str(Walls[2][i])+"\n")

exp.close()

print("fin")

#affichage des murs
ax1.scatter(Walls[0]*tsize+tsize/2,Walls[1]*tsize+tsize/2,Walls[2]*tsize+tsize/2)

ax1.set_xlim3d(0, TailleSim[0]*tsize)
ax1.set_ylim3d(0, TailleSim[1]*tsize)
ax1.set_zlim3d(0, TailleSim[2]*tsize)

plt.show()
#------------------