#include "FLIPimpl.h"

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <helper_cuda.h> 

//différent define pour les Options de la simulation

//#define DEBUG
//#define D_VIT
//#define D_TYPE

//#define D_DIV

#define CFL_FORCED

#define FLIPNESS 0.85

#define RK2
//#define BOUNDARY_WALL_ONLY

#define GRAV
#define GRAVITY 9.81

//#define CENTRAL

#define CENTRAL_F 100
#define CENTRAL_F_X 80.5
#define CENTRAL_F_Y 40.5
#define CENTRAL_F_Z 40.5

#define BUBBLETRSH 0.8

//#define FASTJAC // ne permet pas d'aller plus rapidement

#define SHORTCOMPUTE
//#define LESSW

//------------------

//diverses fonctions qui traitent les indices et quelques fonctions mathématiques

__device__ unsigned int gind(unsigned int indiceX, unsigned int indiceY, unsigned int indiceZ, uint3 BoxIndice)// fonction qui donne les indices grace a une coordonnée 3D
{
	return indiceZ + indiceY * BoxIndice.z + indiceX * BoxIndice.z * BoxIndice.y;
}

__device__ float  gequal(float a)
{
	if (a > 0)
	{
		return a;
	}
	else
	{
		return 0;
	}
}



__device__ float3 interpolate(float3 pos, uint3 box, float3* gridSpeed, float tsize)
{
	unsigned int XGridI = (int)(pos.x / tsize);
	unsigned int YGridI = (int)(pos.y / tsize);
	unsigned int ZGridI = (int)(pos.z / tsize);


	float ax = (pos.x) / tsize - XGridI;
	float ay = (pos.y) / tsize - YGridI;
	float az = (pos.z) / tsize - ZGridI;

	float xvit = (ax)*gridSpeed[gind(XGridI + 1, YGridI, ZGridI, box)].x + (1 - ax) * gridSpeed[gind(XGridI, YGridI, ZGridI, box)].x;
	float yvit = (ay)*gridSpeed[gind(XGridI, YGridI + 1, ZGridI, box)].y + (1 - ay) * gridSpeed[gind(XGridI, YGridI, ZGridI, box)].y;
	float zvit = (az)*gridSpeed[gind(XGridI, YGridI, ZGridI + 1, box)].z + (1 - az) * gridSpeed[gind(XGridI, YGridI, ZGridI, box)].z;

	return make_float3(xvit, yvit, zvit);
}


__device__ float absmin(float x, float limit)
{
	if (fabs(x) < fabs(limit))
	{
		return x;
	}
	else
	{
		return limit * x / fabs(x);
	}
}
//--------------------------

// mettre les vitesses des particules dans la grille
__global__ void TrToGrV2_k(uint3 MACbox, unsigned int partcount, float tsize, float3* MACgridSpeed, float3* MACweight, float3* Ppos, float3* Pvit)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = index; i < partcount; i += stride)
	{
		unsigned int XGridI = (int)(Ppos[index].x / tsize);
		unsigned int YGridI = (int)(Ppos[index].y / tsize);
		unsigned int ZGridI = (int)(Ppos[index].z / tsize);

		float ax = Ppos[index].x / tsize - XGridI;
		float ay = Ppos[index].y / tsize - YGridI;
		float az = Ppos[index].z / tsize - ZGridI;

#ifdef DEBUG

		if (ax < 0 || ax >= 1 || ay < 0 || ay >= 1 || az < 0 || az >= 1) // ne pas oublier de bien centrer les particules
		{
			printf("part : %d mauvais calcul de ax ou ay ou az %f %f %f \n", index, ax, ay, az);
		}

#ifdef D_VIT
		Pvit[index] = make_float3(1, 1, 1);
#endif

#endif

		atomicAdd(&MACgridSpeed[gind(XGridI + 1, YGridI, ZGridI, MACbox)].x, (ax)*Pvit[index].x);
		atomicAdd(&MACgridSpeed[gind(XGridI, YGridI + 1, ZGridI, MACbox)].y, (ay)*Pvit[index].y);
		atomicAdd(&MACgridSpeed[gind(XGridI, YGridI, ZGridI + 1, MACbox)].z, (az)*Pvit[index].z);

		atomicAdd(&MACweight[gind(XGridI + 1, YGridI, ZGridI, MACbox)].x, ax);
		atomicAdd(&MACweight[gind(XGridI, YGridI + 1, ZGridI, MACbox)].y, ay);
		atomicAdd(&MACweight[gind(XGridI, YGridI, ZGridI + 1, MACbox)].z, az);

		atomicAdd(&MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].x, (1 - ax) * Pvit[index].x);
		atomicAdd(&MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].y, (1 - ay) * Pvit[index].y);
		atomicAdd(&MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].z, (1 - az) * Pvit[index].z);

		atomicAdd(&MACweight[gind(XGridI, YGridI, ZGridI, MACbox)].x, (1 - ax));
		atomicAdd(&MACweight[gind(XGridI, YGridI, ZGridI, MACbox)].y, (1 - ay));
		atomicAdd(&MACweight[gind(XGridI, YGridI, ZGridI, MACbox)].z, (1 - az));


	}
}

//transfert de la grille au particule
__global__ void TrToPrV2_k(uint3 MACbox, unsigned int partcount, float tsize, float3* MACgridSpeed, float3* MACweight, float3* Ppos, float3* Pvit, float3* MACgridSpeedSave, float* Pcol)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = index; i < partcount; i += stride)
	{
		unsigned int XGridI = (int)(Ppos[index].x / tsize);
		unsigned int YGridI = (int)(Ppos[index].y / tsize);
		unsigned int ZGridI = (int)(Ppos[index].z / tsize);


		float ax = Ppos[index].x / tsize - XGridI;
		float ay = Ppos[index].y / tsize - YGridI;
		float az = Ppos[index].z / tsize - ZGridI;

		float xvit = (ax)*MACgridSpeed[gind(XGridI + 1, YGridI, ZGridI, MACbox)].x + (1 - ax) * MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].x;
		float yvit = (ay)*MACgridSpeed[gind(XGridI, YGridI + 1, ZGridI, MACbox)].y + (1 - ay) * MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].y;
		float zvit = (az)*MACgridSpeed[gind(XGridI, YGridI, ZGridI + 1, MACbox)].z + (1 - az) * MACgridSpeed[gind(XGridI, YGridI, ZGridI, MACbox)].z;

		float Oxvit = (ax)*MACgridSpeedSave[gind(XGridI + 1, YGridI, ZGridI, MACbox)].x + (1 - ax) * MACgridSpeedSave[gind(XGridI, YGridI, ZGridI, MACbox)].x;
		float Oyvit = (ay)*MACgridSpeedSave[gind(XGridI, YGridI + 1, ZGridI, MACbox)].y + (1 - ay) * MACgridSpeedSave[gind(XGridI, YGridI, ZGridI, MACbox)].y;
		float Ozvit = (az)*MACgridSpeedSave[gind(XGridI, YGridI, ZGridI + 1, MACbox)].z + (1 - az) * MACgridSpeedSave[gind(XGridI, YGridI, ZGridI, MACbox)].z;


#ifdef DEBUG
#ifdef D_VIT
		if ((xvit - Pvit[index].x) != 0 || (yvit - Pvit[index].y) != 0 || (zvit - Pvit[index].z) != 0)
		{
			printf("il y a une difference de : %f %f %f entre la vitesse de base et celle interpole \n", xvit - Pvit[index].x, yvit - Pvit[index].y, zvit - Pvit[index].z);
			//printf("il y a une difference de : %d %f %f %f entre la vitesse de base et celle interpole \n", gind(XGridI + 1, YGridI, ZGridI, MACbox), xvit, yvit, zvit);
		}
#endif
#endif

		float pvitx = xvit * (1 - FLIPNESS) + (xvit - Oxvit + Pvit[index].x) * FLIPNESS;
		float pvity = yvit * (1 - FLIPNESS) + (yvit - Oyvit + Pvit[index].y) * FLIPNESS;
		float pvitz = zvit * (1 - FLIPNESS) + (zvit - Ozvit + Pvit[index].z) * FLIPNESS;

		float dist = pow(pvitx - Pvit[index].x,2) + pow(pvity - Pvit[index].y,2) + pow(pvitz - Pvit[index].z,2);
		float vit = pow(Pvit[index].x,2) + pow(Pvit[index].y,2) + pow(Pvit[index].z,2);

		Pcol[index] = (vit) / 4;


		Pvit[index].x = pvitx;
		Pvit[index].y = pvity;
		Pvit[index].z = pvitz;

	}
}

// normaliser la grille
__global__ void GridNormalV2_k(uint3 MACbox, float3* MACgridSpeed, float3* MACweight, float3* MACgridSpeedSave)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox);
	if (MACweight[index].x != 0)
	{
		MACgridSpeed[index].x = MACgridSpeed[index].x / MACweight[index].x;
		MACgridSpeedSave[index].x = MACgridSpeed[index].x;

	}
	if (MACweight[index].y != 0)
	{
		MACgridSpeed[index].y = MACgridSpeed[index].y / MACweight[index].y;
		MACgridSpeedSave[index].y = MACgridSpeed[index].y;
	}
	if (MACweight[index].z != 0)
	{
		MACgridSpeed[index].z = MACgridSpeed[index].z / MACweight[index].z;
		MACgridSpeedSave[index].z = MACgridSpeed[index].z;
	}

}

//type de cases de fluide met en tant que fluide toute les cases environnante (27 cases)
__global__ void set_typeWater_k(uint3 box, uint3 MACbox, float3* MACweight, unsigned int* type)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (
		(MACweight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x != 0 && MACweight[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x != 0) &&
		(MACweight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y != 0 && MACweight[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, MACbox)].y != 0) &&
		(MACweight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z != 0 && MACweight[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, MACbox)].z != 0))
	{
		type[index] = 2;

		//coté
		if (blockIdx.x >= 1)
		{
			type[gind(blockIdx.x - 1, blockIdx.y, blockIdx.z, box)] = 2;
		}
		if (blockIdx.x <= box.x - 2)
		{
			type[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, box)] = 2;
		}
		if (blockIdx.y >= 1)
		{
			type[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z, box)] = 2;
		}
		if (blockIdx.y <= box.y - 2)
		{
			type[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, box)] = 2;
		}
		if (blockIdx.z >= 1)
		{
			type[gind(blockIdx.x, blockIdx.y, blockIdx.z - 1, box)] = 2;
		}
		if (blockIdx.z <= box.z - 2)
		{
			type[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, box)] = 2;
		}

		// tour central
		if ((blockIdx.z <= box.z - 2) && (blockIdx.y <= box.y - 2))
		{
			type[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z + 1, box)] = 2;
		}

		if ((blockIdx.z <= box.z - 2) && (blockIdx.y >= 1))
		{
			type[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z + 1, box)] = 2;
		}

		if ((blockIdx.z >= 1) && (blockIdx.y <= box.y - 2))
		{
			type[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z - 1, box)] = 2;
		}

		if ((blockIdx.z >= 1) && (blockIdx.y >= 1))
		{
			type[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z - 1, box)] = 2;
		}



		//tour du haut
		if ((blockIdx.z <= box.z - 2) && (blockIdx.y <= box.y - 2) && (blockIdx.x >= 1))
		{
			type[gind(blockIdx.x - 1, blockIdx.y + 1, blockIdx.z + 1, box)] = 2;
		}

		if ((blockIdx.z <= box.z - 2) && (blockIdx.y >= 1) && (blockIdx.x >= 1))
		{
			type[gind(blockIdx.x - 1, blockIdx.y - 1, blockIdx.z + 1, box)] = 2;
		}

		if ((blockIdx.z >= 1) && (blockIdx.y <= box.y - 2) && (blockIdx.x >= 1))
		{
			type[gind(blockIdx.x - 1, blockIdx.y + 1, blockIdx.z - 1, box)] = 2;
		}

		if ((blockIdx.z >= 1) && (blockIdx.y >= 1) && (blockIdx.x >= 1))
		{
			type[gind(blockIdx.x - 1, blockIdx.y - 1, blockIdx.z - 1, box)] = 2;
		}

		//tour bas 
		if ((blockIdx.z <= box.z - 2) && (blockIdx.y <= box.y - 2) && (blockIdx.x <= box.x - 2))
		{
			type[gind(blockIdx.x + 1, blockIdx.y + 1, blockIdx.z + 1, box)] = 2;
		}

		if ((blockIdx.z <= box.z - 2) && (blockIdx.y >= 1) && (blockIdx.x <= box.x - 2))
		{
			type[gind(blockIdx.x + 1, blockIdx.y - 1, blockIdx.z + 1, box)] = 2;
		}

		if ((blockIdx.z >= 1) && (blockIdx.y <= box.y - 2) && (blockIdx.x <= box.x - 2))
		{
			type[gind(blockIdx.x + 1, blockIdx.y + 1, blockIdx.z - 1, box)] = 2;
		}

		if ((blockIdx.z >= 1) && (blockIdx.y >= 1) && (blockIdx.x <= box.x - 2))
		{
			type[gind(blockIdx.x + 1, blockIdx.y - 1, blockIdx.z - 1, box)] = 2;
		}

	}

}

//mise en place des murs
__global__ void set_typeWalls_k(uint3 box, unsigned int* type, float tsize)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (blockIdx.x <= 1 || blockIdx.y <= 1 || blockIdx.z <= 1 || blockIdx.x >= box.x - 2 || blockIdx.y >= box.y - 2 || blockIdx.z >= box.z - 2)
	{
		type[index] = 1;
	}



#ifdef DEBUG
#ifdef D_TYPE
	if (blockIdx.x == 30 && blockIdx.y == 20 && blockIdx.z == 20)
	{
		printf("la case %d %d %d est de type %d \n", blockIdx.x, blockIdx.y, blockIdx.z, type[index]);
	}
#endif
#endif
}

//mur temporaire (pour la simulation banc de fluide)
__global__ void set_typeTempWall_k(uint3 box, unsigned int* type, bool tr, float tsize)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (blockIdx.x * tsize >= 30 + tsize)
	{
		type[index] = 1;
	}

}

__global__ void add_external_forces_k_stg(uint3 box, uint3 MACbox, float3* MACgridSpeed, unsigned int* type, float tstep, float tsize)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox);

	MACgridSpeed[index].y -= GRAVITY * tstep;

}

//ajout de la gravité au case non solide
__global__ void add_external_forces_k(uint3 box, uint3 MACbox, float3* MACgridSpeed, unsigned int* type, float tstep, float tsize)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	//if (type[index] == 2)
	if (type[index] != 1)
	{
		//Gravité

#ifdef GRAV

		MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y -= GRAVITY * tstep;


#endif
		//force Centrale
#ifdef CENTRAL
		if (threadIdx.x == 0)
		{

			float r = pow(blockIdx.x * tsize - CENTRAL_F_X, 2) + pow(blockIdx.y * tsize - CENTRAL_F_Y, 2) + pow(blockIdx.z * tsize - CENTRAL_F_Z, 2);

			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x -= CENTRAL_F * (blockIdx.x * tsize - CENTRAL_F_X) / r * tstep;
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y -= CENTRAL_F * ((blockIdx.y) * tsize - CENTRAL_F_Y) / r * tstep;
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z -= CENTRAL_F * (blockIdx.z * tsize - CENTRAL_F_Z) / r * tstep;
		}

		if (threadIdx.x == 1)
		{

			float r1 = pow((blockIdx.x + 1) * tsize - CENTRAL_F_X, 2) + pow((blockIdx.y + 1) * tsize - CENTRAL_F_Y, 2) + pow((blockIdx.z + 1) * tsize - CENTRAL_F_Z, 2);

			MACgridSpeed[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x -= CENTRAL_F * ((blockIdx.x + 1) * tsize + 1 - CENTRAL_F_X) / r1 * tstep;
			MACgridSpeed[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, MACbox)].y -= CENTRAL_F * ((blockIdx.y + 1) * tsize + 1 - CENTRAL_F_Y) / r1 * tstep;
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, MACbox)].z -= CENTRAL_F * ((blockIdx.z + 1) * tsize - CENTRAL_F_Z) / r1 * tstep;
		}
#endif
	}


}

// intégration leapfrog
__global__ void RKT2_k(int partCount, float3* Ppos, float3* Pvit, float tstep, float tsize, float3* MACGridSpeed, uint3 MACbox, uint3 box, unsigned int* type)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = index; i < partCount; i += stride)
	{


		float intx = Pvit[index].x * tstep / 2 + Ppos[index].x;
		float inty = Pvit[index].y * tstep / 2 + Ppos[index].y;
		float intz = Pvit[index].z * tstep / 2 + Ppos[index].z;

		float3 semivit = interpolate(make_float3(intx, inty, intz), MACbox, MACGridSpeed, tsize);

		unsigned int XGridIB = (int)(Ppos[index].x / tsize);
		unsigned int YGridIB = (int)(Ppos[index].y / tsize);
		unsigned int ZGridIB = (int)(Ppos[index].z / tsize);

#ifdef CFL_FORCED // On force la condition que une particule puisse pas traversé deux cases d'un coup (Conditions de précision)

		Ppos[index].x += absmin(semivit.x * tstep, tsize * 0.95);
		Ppos[index].y += absmin(semivit.y * tstep, tsize * 0.95);
		Ppos[index].z += absmin(semivit.z * tstep, tsize * 0.95);

#else

		Ppos[index].x += semivit.x * tstep;
		Ppos[index].y += semivit.y * tstep;
		Ppos[index].z += semivit.z * tstep;

#endif
		unsigned int XGridI = (int)(Ppos[index].x / tsize);
		unsigned int YGridI = (int)(Ppos[index].y / tsize);
		unsigned int ZGridI = (int)(Ppos[index].z / tsize);

		if (type[gind(XGridI, YGridIB, ZGridIB, box)] == 1 )
		{
			Ppos[index].x = tsize * llroundf(Ppos[index].x / tsize) * (1 + (llroundf(Ppos[index].x / tsize) - (int)(Ppos[index].x / tsize) - 0.5) * 0.05);
			Pvit[index].x = 0;
		}

		if (type[gind(XGridIB, YGridI, ZGridIB, box)] == 1 )
		{
			Ppos[index].y = tsize * llroundf(Ppos[index].y / tsize) * (1 + (llroundf(Ppos[index].y / tsize) - (int)(Ppos[index].y / tsize) - 0.5) * 0.05);
			Pvit[index].y = 0;
		}

		if (type[gind(XGridIB, YGridIB, ZGridI, box)] == 1 )
		{
			Ppos[index].z = tsize * llroundf(Ppos[index].z / tsize) * (1 + (llroundf(Ppos[index].z / tsize) - (int)(Ppos[index].z / tsize) - 0.5) * 0.05);
			Pvit[index].z = 0;
		}

		XGridI = (int)(Ppos[index].x / tsize);
		YGridI = (int)(Ppos[index].y / tsize);
		ZGridI = (int)(Ppos[index].z / tsize);

		if (ZGridI > box.z - 1 || ZGridI < 0 || YGridI > box.y - 1 || YGridI < 0 || XGridI > box.x - 1 || XGridI < 0)
		{
			Pvit[index].x = 0;
			Pvit[index].y = 0;
			Pvit[index].z = 0;
			Ppos[index].x = tsize * 1.1f;
			Ppos[index].y = tsize * 1.1f;
			Ppos[index].z = tsize * 1.1f;

		}

	}

}

//condition au limite de Dirichlet
__global__ void boundaries_k(uint3 box, uint3 MACbox, float3* MACgridSpeed, unsigned int* type)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (type[index] == 1)
	{
		//printf("la case %d %d %d est solide \n", blockIdx.x, blockIdx.y, blockIdx.z);

		MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x = 0;

		MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y = 0;

		MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z = 0;

		MACgridSpeed[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x = 0;

		MACgridSpeed[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, MACbox)].y = 0;

		MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, MACbox)].z = 0;
	}
}

__global__ void pressure_copy_k(uint3 boxind, float* gridpressureB, float* gridpressureA)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, boxind);
	gridpressureB[index] = gridpressureA[index];
}

//calcul de la divergence des vitesses
__global__ void div_calc_k(float3* MACgridSpeed, uint3 box, uint3 MACbox, float* gridDiv, unsigned int* type, float tsize, float3* MACGridWeight, float density, float tstep)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);
	if (type[index] == 2)
	{
		float dens =
			(MACGridWeight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x + MACGridWeight[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x +
				MACGridWeight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y + MACGridWeight[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, MACbox)].y +
				MACGridWeight[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z + MACGridWeight[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, MACbox)].z) / 5;

		gridDiv[index] =
			(MACgridSpeed[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, MACbox)].x - MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x +
				MACgridSpeed[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, MACbox)].y - MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y +
				MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, MACbox)].z - MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z - fmaxf(dens - density, 0))
			* (tsize) / tstep;
	}
}

// algorithme de jacobi
__global__ void jacobi_iter_k(uint3 box, uint3 MACbox, float3* MACgridSpeed, float* gridPressureA, float* gridPressureB, unsigned int* type, float tsize, float* gridDiv)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

#ifdef SHORTCOMPUTE
	if (type[gind(blockIdx.x, blockIdx.y, blockIdx.z, box)] == 2)

	{
#else
	if (type[index] != 1)
	{
#endif

		float div = gridDiv[index];

		int Wc = 0;

		gridPressureA[index] = 0;

		if (type[gind(blockIdx.x - 1, blockIdx.y, blockIdx.z, box)] != 1)
		{
			Wc += 1;
			gridPressureA[index] += gridPressureB[gind(blockIdx.x - 1, blockIdx.y, blockIdx.z, box)];
		}
		if (type[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, box)] != 1)
		{
			Wc += 1;
			gridPressureA[index] += gridPressureB[gind(blockIdx.x + 1, blockIdx.y, blockIdx.z, box)];
		}
		if (type[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z, box)] != 1)
		{
			Wc += 1;
			gridPressureA[index] += gridPressureB[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z, box)];
		}
		if (type[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, box)] != 1)
		{
			Wc += 1;
			gridPressureA[index] += gridPressureB[gind(blockIdx.x, blockIdx.y + 1, blockIdx.z, box)];
		}
		if (type[gind(blockIdx.x, blockIdx.y, blockIdx.z - 1, box)] != 1)
		{
			Wc += 1;
			gridPressureA[index] += gridPressureB[gind(blockIdx.x, blockIdx.y, blockIdx.z - 1, box)];
		}
		if (type[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, box)] != 1)
		{
			Wc += 1;
			gridPressureA[index] += gridPressureB[gind(blockIdx.x, blockIdx.y, blockIdx.z + 1, box)];
		}
		if (Wc != 0)
		{
			gridPressureA[index] -= div;

			gridPressureA[index] /= Wc;
		}
	}
	else
	{
		gridPressureA[index] = 0;
	}
	}

// ajout du gradient de la pression aux vitesses
__global__ void add_pressure_k(uint3 MACbox, uint3 box, float* gridPressure, float3 * MACgridSpeed, float tsize, unsigned int* type, float tstep)
{
	unsigned int index = gind(blockIdx.x, blockIdx.y, blockIdx.z, box);

	if (type[index] == 2)
	{

		if (type[gind(blockIdx.x - 1, blockIdx.y, blockIdx.z, box)] == 2)
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].x -=
				(gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z, box)] - gridPressure[gind(blockIdx.x - 1, blockIdx.y, blockIdx.z, box)]) * tstep / tsize;
		}

		if (type[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z, box)] == 2)
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].y -=
				(gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z, box)] - gridPressure[gind(blockIdx.x, blockIdx.y - 1, blockIdx.z, box)]) * tstep / tsize;
		}

		if (type[gind(blockIdx.x, blockIdx.y, blockIdx.z - 1, box)] == 2)
		{
			MACgridSpeed[gind(blockIdx.x, blockIdx.y, blockIdx.z, MACbox)].z -=
				(gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z, box)] - gridPressure[gind(blockIdx.x, blockIdx.y, blockIdx.z - 1, box)]) * tstep / tsize;
		}
	}
}


// On met en tant que solide les indices qu'il y a dans notre fichier de collision
__global__ void set_collider_k(uint3 box, int3* colliderind,int vecsize,unsigned int* type)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = index; i < vecsize; i += stride)
	{
		type[gind(colliderind[i].x, colliderind[i].z, colliderind[i].y,box)] = 1;
	}
}

// FONCTION externe -----------------------------------------------------------------------------------------------------
// ce n'est uniquement la réecriture des en-tête des fonctions pour permettre le lien entre fonction C++ et fonction en CUDA
extern "C"
void TransfertToGridV2(FlipSim * flipEngine)
{

	unsigned int count = flipEngine->PartCount;

	TrToGrV2_k << <(count / 512 + 1), 512 >> > (
		flipEngine->MACBoxIndice,
		flipEngine->PartCount,
		flipEngine->tileSize,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight,
		flipEngine->Partpos,
		flipEngine->Partvit
		);

	getLastCudaError("Kernel execution failed: TrToGrV2_k");

	uint3 MACbox = flipEngine->MACBoxIndice;

	dim3 MACmat(MACbox.x, MACbox.y, MACbox.z);


	GridNormalV2_k << <MACmat, 1 >> > (
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight,
		flipEngine->MACGridSpeedSave
		);
	getLastCudaError("Kernel execution failed: GridNormalV2_k");

	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);;

	set_typeWater_k << <mat, 1 >> > (
		box,
		MACbox,
		flipEngine->MACGridWeight,
		flipEngine->type);

	getLastCudaError("Kernel execution failed: set_typeWater_k");

	set_typeWalls_k << <mat, 1 >> > (
		box,
		flipEngine->type,
		flipEngine->tileSize);

	int vecsize = (flipEngine->CollideInd).size()/3;
	if (vecsize != 0)
	{
		set_collider_k << <vecsize, 1 >> > (
			box,
			flipEngine->CollideIndCud,
			vecsize,
			flipEngine->type);

		getLastCudaError("Kernel execution failed: set_typeWalls_k");
	}
}


extern "C"
void setTempWall(FlipSim * flipEngine, bool trigger)
{
	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);;

	set_typeTempWall_k << <mat, 1 >> > (
		box,
		flipEngine->type,
		trigger,
		flipEngine->tileSize);

	getLastCudaError("Kernel execution failed: set_typeTempWall_k");

}

extern "C"
void TransfertToPartV2(FlipSim * flipEngine)
{
	unsigned int count = flipEngine->PartCount;

	TrToPrV2_k << <(count / 512 + 1), 512 >> > (
		flipEngine->MACBoxIndice,
		flipEngine->PartCount,
		flipEngine->tileSize,
		flipEngine->MACGridSpeed,
		flipEngine->MACGridWeight,
		flipEngine->Partpos,
		flipEngine->Partvit,
		flipEngine->MACGridSpeedSave,
		flipEngine->Partcol
		);

	getLastCudaError("Kernel execution failed: TrToPrV2_k");

}

extern "C"
void AddExternalForcesV2(FlipSim * flipEngine)
{
	uint3 box = flipEngine->BoxIndice;

	uint3 MACbox = flipEngine->MACBoxIndice;

	dim3 MACmat(MACbox.x, MACbox.y, MACbox.z);

	add_external_forces_k_stg << <MACmat, 1 >> > (
		box,
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->type,
		flipEngine->TimeStep,
		flipEngine->tileSize);

	getLastCudaError("Kernel execution failed: add_external_forces_k");

}

extern "C"
void EulerIntegrateV2(FlipSim * flipEngine)
{
	unsigned int count = flipEngine->PartCount;

#ifdef RK2

	RKT2_k << <(count / 512 + 1), 512 >> > (
		count,
		flipEngine->Partpos,
		flipEngine->Partvit,
		flipEngine->TimeStep,
		flipEngine->tileSize,
		flipEngine->MACGridSpeed,
		flipEngine->MACBoxIndice,
		flipEngine->BoxIndice,
		flipEngine->type);

	getLastCudaError("Kernel execution failed: RKT2_k");

#else

	euler_k << <(count / 512 + 1), 512 >> > (
		count,
		flipEngine->Partpos,
		flipEngine->Partvit,
		flipEngine->TimeStep,
		flipEngine->tileSize);

	getLastCudaError("Kernel execution failed: euler_k");

#endif

}

extern "C"
void BoundariesConditionV2(FlipSim * flipEngine)
{
	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);

	boundaries_k << <mat, 1 >> > (
		box,
		flipEngine->MACBoxIndice,
		flipEngine->MACGridSpeed,
		flipEngine->type);

	getLastCudaError("Kernel execution failed: boundaries_k");
}

extern "C"
void JacobiIterV2(FlipSim * flipEngine, int step)
{
	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);


	div_calc_k << <mat, 1 >> > (
		flipEngine->MACGridSpeed,
		box,
		flipEngine->MACBoxIndice,
		flipEngine->GridDiv,
		flipEngine->type,
		flipEngine->tileSize,
		flipEngine->MACGridWeight,
		12.0 ,
		flipEngine->TimeStep);

	getLastCudaError("Kernel execution failed: div_calc_k");

	for (int i = 0; i < step; i++)
	{

		jacobi_iter_k << <mat, 1 >> > (
			box,
			flipEngine->MACBoxIndice,
			flipEngine->MACGridSpeed,
			flipEngine->GridPressureA,
			flipEngine->GridPressureB,
			flipEngine->type,
			flipEngine->tileSize,
			flipEngine->GridDiv);

		getLastCudaError("Kernel execution failed: jacobi_iter_k");

		pressure_copy_k << <mat, 1 >> > (
			box,
			flipEngine->GridPressureB,
			flipEngine->GridPressureA);

		getLastCudaError("Kernel execution failed: pressure_copy_k");
	}


}

extern "C"
void AddPressureV2(FlipSim * flipEngine)
{
	uint3 box = flipEngine->BoxIndice;
	dim3 mat(box.x, box.y, box.z);

	add_pressure_k << <mat, 1 >> > (
		flipEngine->MACBoxIndice,
		box,
		flipEngine->GridPressureB,
		flipEngine->MACGridSpeed,
		flipEngine->tileSize,
		flipEngine->type,
		flipEngine->TimeStep);

	getLastCudaError("Kernel execution failed: add_pressure_k");
}
//-----------------------------