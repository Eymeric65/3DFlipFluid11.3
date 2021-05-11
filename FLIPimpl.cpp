#include "FLIPimpl.h"
#include <assert.h>



extern "C" void TransfertToGridV2(FlipSim * flipEngine);

extern "C" void TransfertToPartV2(FlipSim * flipEngine);

extern "C" void AddExternalForcesV2(FlipSim * flipEngine);

extern "C" void EulerIntegrateV2(FlipSim * flipEngine);

extern "C" void BoundariesConditionV2(FlipSim * flipEngine);

extern "C" void JacobiIterV2(FlipSim * flipEngine, int step);

extern "C" void AddPressureV2(FlipSim * flipEngine);

extern "C" void setTempWall(FlipSim * flipEngine, bool trigger);

FlipSim::FlipSim(float width, float height,float length, float tsize, unsigned int partcount,float tstep, std::ifstream &collider )
{

	PartCount = partcount;

	TimeStep = tstep;

	Positions = (float3*)malloc(sizeof(float3) * PartCount);

	std::cout << "reset partvit" << std::endl;
	//cudaMalloc(&pos, PartCount * sizeof(float3));
	std::string line;

	if (collider.is_open())
	{
		while (std::getline(collider, line))
		{

			int ind [3] ;

			std::string delimiter = " ";

			size_t pos = 0;
			//std::string token;
			//std::cout << "ah" << std::endl;
			while ((pos = line.find(delimiter)) != std::string::npos) {
				CollideInd.push_back(stoi(line.substr(0, pos)));
				//std::cout << stoi(line.substr(0, pos))  << std::endl;
				line.erase(0, pos + delimiter.length());
			}
			CollideInd.push_back(stoi(line.substr(pos + delimiter.length())));
			//std::cout << stoi(line.substr(pos + delimiter.length())) << std::endl;
			//std::cout << ind[0] << std::endl;
		}
		collider.close();
	}

	//std::cout << CollideInd[800] << std::endl;

	BoxSize = make_float3(width, height, length);

	//positions = 2;

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

	cudaMalloc(&CollideIndCud, CollideInd.size() * sizeof(int));
	cudaMemcpy(CollideIndCud, &CollideInd[0], CollideInd.size() * sizeof(int), cudaMemcpyHostToDevice);



}

void FlipSim::TransferToGrid()
{
	cudaMemset(MACGridWeight, 0, MACIndiceCount * sizeof(float3));
	cudaMemset(MACGridSpeed, 0, MACIndiceCount * sizeof(float3));
	cudaMemset(type,0, IndiceCount * sizeof(unsigned int));
	cudaMemset(GridDiv, 0, IndiceCount * sizeof(float));
	//std::cout << " coucou " << std::endl;

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

	JacobiIterV2(this, 100);

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
	cudaFree(MACGridSpeed);
	cudaFree(MACGridWeight);
	cudaFree(type);
	cudaFree(GridPressureB);
	cudaFree(GridPressureA);

	cudaFree(Partvit);
}

void FlipSim::StartCompute()
{
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
	cudaMemcpy( Positions,Partpos, PartCount * sizeof(float3), cudaMemcpyDeviceToHost);

	cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0);

	cudaGraphicsUnmapResources(1, &cuda_col_resource, 0);
	//std::cout << "la taille est " << Partpos[0].x << std::endl;

	

}

void FlipSim::Boundaries()
{
	BoundariesConditionV2(this);
}