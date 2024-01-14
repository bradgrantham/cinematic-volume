#version 450

layout (location = 0) in vec3 input_position;
layout (location = 1) in vec3 input_normal;
layout (location = 2) in vec4 input_color;
layout (location = 3) in vec2 input_texcoord;

layout (location = 0) out vec4 output_position;
layout (location = 1) out vec3 output_normal;
layout (location = 2) out vec4 output_color;
layout (location = 3) out vec2 output_texcoord;

layout (binding = 0) uniform UBO 
{
    mat4 modelview;
    mat4 modelview_normal;
    mat4 projection;
} ubo;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main() 
{
   gl_Position =  ubo.projection * ubo.modelview * vec4(input_position.xyz, 1.0);
   output_position =  ubo.modelview * vec4(input_position.xyz, 1.0);
   output_normal = (ubo.modelview_normal * vec4(input_normal, 0.0)).xyz;
   output_color = input_color;
   output_texcoord = input_texcoord;
}
