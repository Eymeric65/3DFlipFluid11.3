#ifndef FLIP_H
#define FLIP_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>


#include <fstream>
#include <string>
#include <vector>

class FlipSim
{

public:

	float3 BoxSize;
	float tileSize;
	
	float3* Positions;

	uint3 BoxIndice;

	uint3 MACBoxIndice;
	int IndiceCount;
	int MACIndiceCount;

	float3* MACGridSpeed;
	float3* MACGridSpeedSave;

	float3* MACGridWeight;

	unsigned int* type; // 0 air 1 solide 2 fluide

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

	std::vector<int> CollideInd;

	int3* CollideIndCud; 

	FlipSim(float width, float height, float length, float tsize, unsigned int partcount, float tstep, std::ifstream &scollider);

	void TransferToGrid();

	void TransferToParticule();

	void AddExternalForces();

	void Integrate();

	void StartCompute();

	void EndCompute();

	void linkPos(GLuint buffer);

	void linkCol(GLuint buffer);

	void Boundaries();

	void PressureCompute();

	void AddPressure();

	void TempWalls(bool Trigger);
	
	void endSim();
};

#endif