#pragma once

#include <vector>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

int spherecreate(std::vector<GLfloat>& vertices, std::vector<GLuint>& indices, int lats, int longs)
{
    int i, j;

    int verticesC = 0;

    for (i = 0; i <= lats; i++) {

        float lat0 = glm::pi<float>() * (-0.5 + (float)(i) / (lats));
        float z0 = sin(lat0);
        float zr0 = cos(lat0);

        //std::cout << i << std::endl ; 

        //float lat1 = glm::pi<float>() * (-0.5 + (float)i / lats);
        //float z1 = sin(lat1);
        //float zr1 = cos(lat1);

        int k1 = i * (longs + 1);
        int k2 = (i + 1) * (longs + 1);

        for (j = 0; j <= longs; j++, k1++, k2++) {
            float lng = 2 * glm::pi<float>() * (float)(j) / longs;
            float x = cos(lng);
            float y = sin(lng);


            vertices.push_back(x * zr0);
            vertices.push_back(y * zr0);
            vertices.push_back(z0);

            //indices.push_back(actualindice);

            if (i != 0)
            {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);

                verticesC += 3;
            }

            if (i != (lats - 1))
            {

                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);

                verticesC += 3;
            }

        }

    }

    return verticesC;

}