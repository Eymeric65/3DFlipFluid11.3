#Ce script permet d'afficher les points de la simulation sur blender pour effectuer ensuite le rendu

import bpy 
import os
import numpy as np
import struct
import mmap

FileName = "D:/bureau/prepa/TIPE/cuda 11.3/3DFlipFluid11.3/Result/Simulate_01_b.dat" #Fichier de simulation

def particleSetter(self,degp):
    
    
    Result = open(FileName,"rb")

    PartCount = int.from_bytes(Result.read(4), byteorder='little', signed=True) #on extrait le nombre de particule
    
    mRes = mmap.mmap(Result.fileno(), 0,access=mmap.ACCESS_READ) #on crée l'array mmap
    
    Result.close()
    
    #on accede a l'array des pointeur des particules
    particle_systems = object.evaluated_get(degp).particle_systems
    particles = particle_systems[0].particles
    totalParticles = len(particles)
    #-------------------------------------------
    
    #on accède aux propriétés de la scène blender
    scene = bpy.context.scene
    cFrame = scene.frame_current
    sFrame = scene.frame_start

    #on écrase les données précédentes
    if cFrame == sFrame:
        psSeed = object.particle_systems[0].seed
        object.particle_systems[0].seed = psSeed

    #pour les images qui sont compatible avec la taille de la simulation on actualise les valeurs des position
    if (cFrame >= 2 ):
        
        particles.foreach_set("location", struct.unpack('f'*3*PartCount, mRes[((cFrame-2)*3*4)*PartCount + 4:((cFrame-2)*3*4 + 12)*PartCount + 4]))
    
    



Result = open(FileName,"rb")    
PartCount = int.from_bytes(Result.read(4), byteorder='little', signed=True)
#mRes = mmap.mmap(Result.fileno(), 0,access=mmap.ACCESS_READ) #on crée l'array mmap
print(PartCount)
Result.close()

#initialisation de l'instance de particule dans blender 
object = bpy.data.objects["Cube"]
object.modifiers.new("ParticleSystem", 'PARTICLE_SYSTEM')
object.particle_systems[0].settings.count = PartCount
object.particle_systems[0].settings.frame_start = 1
object.particle_systems[0].settings.frame_end = 1
object.particle_systems[0].settings.lifetime = 1000
object.show_instancer_for_viewport = False
#degp = bpy.context.evaluated_depsgraph_get()
#-----------------------------------------------------

#on reset toute fonction qui s'exécute au changement de frame
bpy.app.handlers.frame_change_post.clear()

#on met notre fonction d'update dans l'handler
bpy.app.handlers.frame_change_post.append(particleSetter)

#on actualise la frame a l'image de départ
bpy.context.scene.frame_current = 2