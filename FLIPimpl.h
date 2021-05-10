#ifndef FLIP_H
#define FLIP_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

class FlipSim
{

public:

	//float PartpositionsK;

	float3 BoxSize;
	float tileSize;
	
	float time2;

	uint3 BoxIndice;

	uint3 MACBoxIndice;
	int IndiceCount;
	int MACIndiceCount;

	float3* MACGridSpeed;
	float3* MACGridSpeedSave;

	float3* MACGridWeight;

	unsigned int* type; // 0 is air 1 is solid 2 is fluid

	float* GridPressureB;
	float* GridPressureA;

	float* GridDiv;

	int PartCount;

	float3* Partpos;
	
	float3* Partvit;


	float* Partcol;

	float TimeStep;

	struct cudaGraphicsResource* cuda_pos_resource;
	size_t num_bytes_pos;

	struct cudaGraphicsResource* cuda_col_resource;
	size_t num_bytes_col;

	FlipSim(float width, float height, float length, float tsize, unsigned int partcount, float tstep);

	void TransferToGrid();

	void TransferToParticule();

	void AddExternalForces();

	void Integrate(); //euler

	void StartCompute();

	void EndCompute();

	void linkPos(GLuint buffer);

	void linkCol(GLuint buffer);

	void Boundaries();

	void PressureCompute();

	void AddPressure();

	void TempWalls(bool Trigger);

	//a faire
	
	void endSim();




};



#endif