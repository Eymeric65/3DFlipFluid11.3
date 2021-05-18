#version 330 core
layout (location = 0) in vec3 aPos;

layout (location = 1) in vec3 ainstancepos;

layout (location = 2) in float bubbleintensity;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 Normal;
out vec3 FragPos;
out float bubbleintense;

void main()
{
   FragPos = vec3(model * vec4(aPos, 1.0));
   gl_Position = projection * view * (model * vec4(aPos, 1.0) + vec4(ainstancepos,1.0));
   Normal = aPos ;
   bubbleintense = bubbleintensity;
}