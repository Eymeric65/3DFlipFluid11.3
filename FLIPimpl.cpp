#include "FLIPimpl.h"
#include <assert.h>

#define COLLIDER

extern "C" void TransfertToGridV2(FlipSim * flipEngine);

extern "C" void TransfertToPartV2(FlipSim * flipEngine);

extern "C" void AddExternalForcesV2(FlipSim * flipEngine);

extern "C" void EulerIntegrateV2(FlipSim * flipEngine);

extern "C" void BoundariesConditionV2(FlipSim * flipEngine);

extern "C" void JacobiIterV2(FlipSim * flipEngine, int step);

extern "C" void AddPressureV2(FlipSim * flipEngine);

extern "C" void setTempWall(FlipSim * flipEngine, bool trigger);

//création de la classe 
FlipSim::FlipSim(float width, float height,float length, float tsize, unsigned int partcount,float tstep, std::ifstream &collider )
{

	PartCount = partcount;

	TimeStep = tstep;

	Positions = (float3*)malloc(sizeof(float3) * PartCount);

	//extraction des indices des cases pour les mettre dans un tableaux
#ifdef COLLIDER
	if (collider.is_open())
	{
		std::string line;
		while (std::getline(collider, line))
		{

			int ind [3] ;

			std::string delimiter = " ";

			size_t pos = 0;

			while ((pos = line.find(delimiter)) != std::string::npos) {
				CollideInd.push_back(stoi(line.substr(0, pos)));
				line.erase(0, pos + delimiter.length());
			}
			CollideInd.push_back(stoi(line.substr(pos + delimiter.length())));

		}
		collider.close();
	}

	std::cout << "Collider loaded : " << CollideInd.size() << std::endl;

#endif

	
	//--------------

	BoxSize = make_float3(width, height, length);

	tileSize = tsize;

	BoxIndice = make_uint3((int)(BoxSize.x / tileSize),
							(int)(BoxSize.y/ tileSize),
							(int)(BoxSize.z / tileSize)  );

	assert( "la taille de la case n'est pas un multiple de la taille de la boite"&&
		    ((BoxSize.x / tileSize) == BoxIndice.x) &&
			((BoxSize.y / tileSize) == BoxIndice.y) &&
			((BoxSize.z / tileSize) == BoxIndice.z)
			);

	MACBoxIndice = make_uint3((int)(BoxSize.x / tileSize)+1,
		(int)(BoxSize.y / tileSize)+1,
		(int)(BoxSize.z / tileSize)+1);


	IndiceCount = BoxIndice.x * BoxIndice.y * BoxIndice.z;

	MACIndiceCount = (BoxIndice.x +1)* (BoxIndice.y+1) *( BoxIndice.z+1);

	printf("il y a %d cases \n",IndiceCount);

	printf("la boite possède (%d;%d;%d)cases \n", BoxIndice.x, BoxIndice.y, BoxIndice.z);

	printf("il y a %d particules \n", PartCount);

	//Allocation mémoire
	cudaMalloc(&Partvit, PartCount * sizeof(float3));
	cudaMemset(Partvit, 0, PartCount * sizeof(float3));

	cudaMalloc(&MACGridSpeedSave, MACIndiceCount * sizeof(float3));

	cudaMalloc(&MACGridSpeed, (MACIndiceCount) * sizeof(float3));//
	cudaMemset(MACGridSpeed, 0, (MACIndiceCount) * sizeof(float3));

	cudaMalloc(&MACGridWeight, MACIndiceCount * sizeof(float3));//
	cudaMemset(MACGridWeight, 0, MACIndiceCount * sizeof(float3));

	cudaMalloc(&GridPressureB, IndiceCount * sizeof(float)); //
	cudaMemset(GridPressureB, 0, IndiceCount * sizeof(float));

	cudaMalloc(&GridPressureA, IndiceCount * sizeof(float));

	cudaMalloc(&type, IndiceCount * sizeof(unsigned int)); //

	cudaMalloc(&GridDiv, IndiceCount * sizeof(float));

#ifdef COLLIDER
	cudaMalloc(&CollideIndCud, CollideInd.size() * sizeof(int));
	cudaMemcpy(CollideIndCud, &CollideInd[0], CollideInd.size() * sizeof(int), cudaMemcpyHostToDevice);
#endif
	//--------------
}

void FlipSim::TransferToGrid()
{
	cudaMemset(MACGridWeight, 0, MACIndiceCount * sizeof(float3));
	cudaMemset(MACGridSpeed, 0, MACIndiceCount * sizeof(float3));
	cudaMemset(type,0, IndiceCount * sizeof(unsigned int));
	cudaMemset(GridDiv, 0, IndiceCount * sizeof(float));

	TransfertToGridV2(this);
}

void FlipSim::TransferToParticule()
{
	TransfertToPartV2(this);
}

void FlipSim::AddExternalForces()
{
	AddExternalForcesV2(this);
}

void FlipSim::PressureCompute()
{
	cudaMemset(GridPressureB, 0, IndiceCount * sizeof(float));

	JacobiIterV2(this, 100); // 100 itérations de Jacobi
}

void FlipSim::AddPressure()
{
	AddPressureV2(this);
}

void FlipSim::TempWalls(bool Trigger)
{
	setTempWall(this,Trigger);
}

void FlipSim::endSim()
{
	//libération mémoire
	cudaFree(MACGridSpeed);
	cudaFree(MACGridWeight);
	cudaFree(type);
	cudaFree(GridPressureB);
	cudaFree(GridPressureA);
	cudaFree(Partvit);
}

void FlipSim::StartCompute()
{
	//lie le buffer graphique et le buffer CUDA
	cudaGraphicsMapResources(1, &cuda_pos_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&Partpos, &num_bytes_pos, cuda_pos_resource);

	cudaGraphicsMapResources(1, &cuda_col_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&Partcol, &num_bytes_col, cuda_col_resource);

}

void FlipSim::linkPos(GLuint buffer)
{
	cudaGraphicsGLRegisterBuffer(&cuda_pos_resource, buffer, cudaGraphicsRegisterFlagsNone);
}

void FlipSim::linkCol(GLuint buffer)
{
	cudaGraphicsGLRegisterBuffer(&cuda_col_resource, buffer, cudaGraphicsRegisterFlagsNone);
}

void FlipSim::Integrate()
{
	EulerIntegrateV2(this);
}

void FlipSim::EndCompute()
{
	// Copie des données de position dans un tableaux du CPU
	cudaMemcpy( Positions,Partpos, PartCount * sizeof(float3), cudaMemcpyDeviceToHost);

	//libération des données
	cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0);
	cudaGraphicsUnmapResources(1, &cuda_col_resource, 0);

}

void FlipSim::Boundaries()
{
	BoundariesConditionV2(this);
}