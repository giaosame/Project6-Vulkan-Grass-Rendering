#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraBufferObject 
{
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) in vec3 fs_nor;
layout(location = 0) out vec4 outColor;

vec3 lightDir = vec3(5, 2, 5);
vec3 green = vec3(max(18, 100 * sin(fs_nor.x)), 250, max(18, 100 * cos(fs_nor.x))) / 255.0;

void main() 
{
    // Compute fragment color
    float diffuseTerm = dot(normalize(fs_nor), normalize(lightDir));
	diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
   
    float ambientTerm = 0.25;
    float lightIntensity = diffuseTerm + ambientTerm; 
	outColor = vec4(lightIntensity * green, 1.0);
}
