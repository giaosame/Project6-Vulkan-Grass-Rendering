#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 0) uniform ModelBufferObject 
{
    mat4 model;
};

layout(location = 0) in vec4 v0;
layout(location = 1) in vec4 v1;
layout(location = 2) in vec4 v2;
layout(location = 3) in vec4 up;
layout(location = 0) out vec4 v0_out;
layout(location = 1) out vec4 v1_out;
layout(location = 2) out vec4 v2_out;
layout(location = 3) out vec4 up_out;

out gl_PerVertex 
{
    vec4 gl_Position;
};

void main() 
{
    v0_out = gl_Position = model * vec4(vec3(v0), 1.0);
	v0_out.w = v0.w;
	v1_out = model * vec4(vec3(v1), 1.0);
	v1_out.w = v1.w;
	v2_out = model * vec4(vec3(v2), 1.0);
	v2_out.w = v2.w; 
	up_out = model * vec4(vec3(up), 0.0);
	up_out.w = up.w;
}
