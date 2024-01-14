#version 450

layout (binding = 1) uniform UBO2
{
    vec3 light_position;
    vec3 light_color;
} ubo2;

layout (binding = 2) uniform Shading
{
    vec3 specular_color;
    float shininess;
} shading;

layout (binding = 3) uniform sampler2D color_texture;

layout (location = 0) in vec4 vertex_position;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec4 vertex_color;
layout (location = 3) in vec2 vertex_texcoord;

layout (location = 0) out vec4 outFragColor;

vec3 unitvec(vec4 p1, vec4 p2)
{
    if(p1.w == 0 && p2.w == 0) {
        return vec3(p2 - p1);
    }
    if(p1.w == 0) {
        return vec3(-p1);
    }
    if(p2.w == 0) {
        return vec3(p2);
    }
    return p2.xyz / p2.w - p1.xyz / p1.w;
}

void main()
{
    vec3 normal = normalize(vertex_normal);
    vec3 eye_position = -vertex_position.xyz; 
    vec3 edir = normalize(eye_position);
    if(dot(normal, edir) < 0) {
        normal *= -1;
    }
    vec3 ldir = normalize(ubo2.light_position - vertex_position.xyz);
    vec3 refl = reflect(-ldir, normal);
    vec3 diffuse = max(0, dot(normal, ldir)) * ubo2.light_color * texture(color_texture, vertex_texcoord).xyz;
    vec3 ambient = ubo2.light_color;
    // vec4 specular = pow(max(0, dot(refl, edir)), material_shininess) * light_color * .8; // from spin
    vec3 specular = pow(max(0, dot(refl, edir)), shading.shininess) * shading.specular_color * ubo2.light_color;
    vec3 material_diffuse = vec3(.8, .8, .8);
    vec3 material_ambient = vec3(.1, .1, .1);
    vec3 material_specular = vec3(.8, .8, .8);
    // specular = vec4(0.0, 0.0, 0.0, 0.0); // from spin
    // color = diffuse * material_diffuse * vertex_color + ambient * material_ambient * vertex_color + specular * material_specular; // from spin
    outFragColor = vec4(vertex_color.xyz * (diffuse * material_diffuse + ambient * material_ambient + specular * material_specular ), 1.0);
}
