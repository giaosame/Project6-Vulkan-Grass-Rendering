#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(quads, equal_spacing, ccw) in;

layout(set = 0, binding = 0) uniform CameraBufferObject 
{
    mat4 view;
    mat4 proj;
	vec3 pos;
} camera;

layout(location = 0) patch in vec4 v0_tese;
layout(location = 1) patch in vec4 v1_tese;
layout(location = 2) patch in vec4 v2_tese;
layout(location = 3) patch in vec4 up_tese;

layout(location = 0) out vec3 fs_nor;

void main()
{
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

	vec3 v0 = gl_in[0].gl_Position.xyz;
	vec3 v1 = v1_tese.xyz;
	vec3 v2 = v2_tese.xyz;

	// De-caltesjaul
	vec3 a = v0 + v * (v1 - v0);
	vec3 b = v1 + v * (v2 - v1);
	vec3 c = a + v * (b - a);
	
	float ori = v0_tese.w; // orientation
	float wid = v2_tese.w; // width

	float dirx = cos(ori);
	float dirz = sin(ori);
	vec3 t1 = vec3(dirx, 0, dirz);
	vec3 c0 = c - wid * t1;
	vec3 c1 = c + wid * t1;

	vec3 t0 = normalize(b - a);
	fs_nor = normalize(cross(t0, t1));
	
	float t = u - u * v; 
	vec3 p = (1.0 - t) * c0 + t * c1;
	
	gl_Position = camera.proj * camera.view * vec4(p, 1);
}
