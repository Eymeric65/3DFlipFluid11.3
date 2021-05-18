#version 330 core
out vec4 FragColor;

uniform vec3 lightColor;
uniform vec3 objectColor;
uniform vec3 lightPos;

in vec3 Normal;
in vec3 Fragpos;
in float bubbleintense;

void main()
{

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - Fragpos);  

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;

    vec3 result = (ambient +diffuse)* (objectColor * (0.85+bubbleintense) );
    FragColor = vec4(result, 1.0);

}