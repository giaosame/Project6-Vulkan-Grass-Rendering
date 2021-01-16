#version 450
#extension GL_ARB_separate_shader_objects : enable
layout(vertices = 1) out;

layout(set = 0, binding = 0) uniform CameraBufferObject 
{
    mat4 view;
    mat4 proj;
	vec3 pos;
} camera;

layout(location = 0) in vec4 v0[];
layout(location = 1) in vec4 v1[];
layout(location = 2) in vec4 v2[];
layout(location = 3) in vec4 up[];
layout(location = 0) patch out vec4 v0_tese;
layout(location = 1) patch out vec4 v1_tese;
layout(location = 2) patch out vec4 v2_tese;
layout(location = 3) patch out vec4 up_tese;

void main()
{
	// Don't move the origin location of the patch
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

	v0_tese = v0[gl_InvocationID];
	v1_tese = v1[gl_InvocationID];
	v2_tese = v2[gl_InvocationID];
	up_tese = up[gl_InvocationID];

	// Tess depend on the depth of origin location
	vec4 pos = vec4(gl_in[gl_InvocationID].gl_Position.xyz, 1.0);
	pos = camera.proj * camera.view * pos;
	vec3 ndcPos = pos.xyz / pos.w;
	float depth = clamp(ndcPos.z, 0.0, 1.0);

	int max_level = 3;
	int level = max_level - int(depth / 0.5);
	level = max(1, level);

	// for bottom and up
	int max_level2 = 2;
	int level2 = max_level2 - int(depth / 0.5); 
	level2 = max(1, level2);

    gl_TessLevelInner[0] = level2;
    gl_TessLevelInner[1] = level2;

    gl_TessLevelOuter[0] = level;
    gl_TessLevelOuter[1] = level2;
    gl_TessLevelOuter[2] = level;
    gl_TessLevelOuter[3] = level2;
}

