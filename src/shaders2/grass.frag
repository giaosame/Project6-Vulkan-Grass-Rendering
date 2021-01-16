#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraBufferObject 
{
    mat4 view;
    mat4 proj;
	//vec3 pos;
} camera;

layout(location = 0) in vec3 fs_nor;
layout(location = 0) out vec4 outColor;

vec3 fs_LightVec = vec3(0.1, 0.3, 0.1);
vec3 baseCol = vec3(25.0, 255.0, 25.0) / 255.;

void main() 
{
	
    // TODO: Compute fragment color
    float diffuseTerm = dot(normalize(fs_nor), normalize(fs_LightVec));
    
	diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
   
    float ambientTerm = 0.2;
    float lightIntensity = diffuseTerm + ambientTerm; 

    vec3 temp = vec3(baseCol.rgb * lightIntensity);
	outColor = vec4(temp, 1.0);
    //outColor = vec4(1.0);
}
