/*
   Copyright 2018-2021 Brad Grantham
   Portions copyright 2018 Jesse Barker
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef __VECTORMATH_H__
#define __VECTORMATH_H__

#include <cmath>
#include <limits>
#include <algorithm>

/*
Style to do:
    vec{234}{i,f} should be specialized from std::array?
    size_t, not int, in arrays?
    run through clang-format to fix single-line, no-brace code
*/

namespace vectormath
{

constexpr float pi = 3.14159265358979323846f;

};

template <class V, int D=V::vector_size>
float dot(const V& v0, const V& v1)
{
    float total = 0;
    for(int i = 0; i < D; i++) { total += v0[i] * v1[i]; }
    return total;
}

template <class V, int D=V::vector_size>
V min(const V& v0, const V& v1)
{
    V tmp;
    for(int i = 0; i < D; i++) { tmp[i] = std::min(v0[i], v1[i]); }
    return tmp;
}

template <class V, int D=V::vector_size>
V max(const V& v0, const V& v1)
{
    V tmp;
    for(int i = 0; i < D; i++) { tmp[i] = std::max(v0[i], v1[i]); }
    return tmp;
}

template <class V, int D=V::vector_size>
float length(const V& v)
{
    float sum = 0;
    for(int i = 0; i < D; i++) { sum += v[i] * v[i]; }
    return sqrtf(sum);
}

template <class V, int D=V::vector_size>
float length_sq(const V& v)
{
    float sum = 0;
    for(int i = 0; i < D; i++) { sum += v[i] * v[i]; }
    return sum;
}

template <class V, int D=V::vector_size>
V normalize(const V& v)
{
    float l = length(v);
    V tmp;
    for(int i = 0; i < D; i++) { tmp[i] = v[i] / l; }
    return tmp;
}

/* normalized i, n */
/* doesn't normalize r */
/* r = u - 2 * n * dot(n, u) */
template <class V, int D=V::vector_size>
V reflect(const V& i, const V& n)
{
    return i - 2.0f * n * dot(i, n);
}

template <class V, int D=V::vector_size>
bool refract(float eta, const V& i, const V& n, V& result)
{
    float ndoti = dot(n, i); 

    float k = 1.0f - eta * eta * (1.0f - ndoti * ndoti);

    if(k < 0.0f) {
        result = V(0, 0, 0);
        return true;
    }

    result = eta * i - (eta * ndoti + sqrtf(k)) * n;
    return false;
}

template <class V, int D=V::vector_size>
bool operator==(const V& v0, const V& v1)
{
    V tmp;
    for(int i = 0; i < D; i++)
        if(v0[i] != v1[i])
            return false;
    return true;
}
    
template <class V, int D=V::vector_size>
V operator+(const V& v0, const V& v1)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v0[i] + v1[i];
    return tmp;
}

template <class V, int D=V::vector_size>
V operator-(const V& v0, const V& v1)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v0[i] - v1[i];
    return tmp;
}

template <class V, int D=V::vector_size>
V operator+(const V& v0, float v1)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v0[i] + v1;
    return tmp;
}

template <class V, int D=V::vector_size>
V operator-(const V& v0, float v1)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v0[i] - v1;
    return tmp;
}

template <class V, int D=V::vector_size>
V operator-(const V& v)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = -v[i];
    return tmp;
}

template <class V, int D=V::vector_size>
V operator*(float w, const V& v) 
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v[i] * w;
    return tmp;
}

template <class V, int D=V::vector_size>
V operator/(float w, const V& v)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v[i] / w;
    return tmp;
}

template <class V, int D=V::vector_size>
V operator*(const V& v, float w)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v[i] * w;
    return tmp;
}

template <class V, int D=V::vector_size>
V operator/(const V& v, float w)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v[i] / w;
    return tmp;
}

template <class V, int D=V::vector_size>
V operator*(const V& v0, const V& v1)
{
    V tmp;
    for(int i = 0; i < D; i++) tmp[i] = v0[i] * v1[i];
    return tmp;
}

struct vec2
{
    float m_v[2];
    static constexpr int vector_size = 2;
    typedef float comp_type;

    vec2(void)
    { }

    vec2(float x_, float y_)
    { set(x_, y_); }

    void set(float x_, float y_)
        { m_v[0] = x_; m_v[1] = y_; }

    vec2(float v)
    {
	for(int i = 0; i < 2; i++) m_v[i] = v;
    }

    vec2(const float *v)
    {
	for(int i = 0; i < 2; i++) m_v[i] = v[i];
    }

    vec2(const vec2 &v)
    {
	for(int i = 0; i < 2; i++) m_v[i] = v[i];
    }

    vec2 &operator=(const float *v) {
	for(int i = 0; i < 2; i++) m_v[i] = v[i];
	return *this;
    }

    vec2 &operator=(const vec2& v) {
	for(int i = 0; i < 2; i++) m_v[i] = v[i];
	return *this;
    }

    vec2(float *v)
        { set(v); }

    operator const float*() const { return m_v; }
    operator float*() { return m_v; }

    float& operator[] (int i)
        { return m_v[i]; }

    const float& operator[] (int i) const
        { return m_v[i]; }

    void clear() { 
	for(int i = 0; i < 2; i++) m_v[i] = 0;
    }

    void set(const float *v)
	{ for(int i = 0; i < 2; i++) m_v[i] = v[i]; }

    vec2 operator*=(float w) {
	for(int i = 0; i < 2; i++) m_v[i] *= w;
	return *this;
    }

    vec2 operator/=(float w) {
	for(int i = 0; i < 2; i++) m_v[i] /= w;
	return *this;
    }

    vec2 operator+=(const vec2& v) {
	for(int i = 0; i < 2; i++) m_v[i] += v[i];
	return *this;
    }

    vec2 operator-=(const vec2& v) {
	for(int i = 0; i < 2; i++) m_v[i] -= v[i];
	return *this;
    }

    float x() const { return m_v[0]; }
    float y() const { return m_v[1]; }
};

struct vec3
{
    float m_v[3];
    static constexpr int vector_size = 3;
    typedef float comp_type;

    vec3(void)
    { }

    void set(float x_, float y_, float z_)
        { m_v[0] = x_; m_v[1] = y_; m_v[2] = z_;}

    vec3(float x_, float y_, float z_)
        { set(x_, y_, z_); }

    vec3(float v)
    {
	for(int i = 0; i < 3; i++) m_v[i] = v;
    }

    vec3(const float *v)
    {
	for(int i = 0; i < 3; i++) m_v[i] = v[i];
    }

    vec3(const vec3 &v) 
    {
	for(int i = 0; i < 3; i++) m_v[i] = v[i];
    }

    vec3 &operator=(const vec3& v) {
	for(int i = 0; i < 3; i++) m_v[i] = v[i];
	return *this;
    }

    vec3 &operator=(float v) {
	for(int i = 0; i < 3; i++) m_v[i] = v;
	return *this;
    }

    vec3(float *v)
    { set(v); }

    operator const float*() const { return m_v; }
    operator float*() { return m_v; }

    float& operator[] (int i)
        { return m_v[i]; }

    const float& operator[] (int i) const
        { return m_v[i]; }

    void clear() { 
	for(int i = 0; i < 3; i++) m_v[i] = 0;
    }

    void set(const float *v)
	{ for(int i = 0; i < 3; i++) m_v[i] = v[i]; }


    vec3 operator*=(float w_) {
	for(int i = 0; i < 3; i++) m_v[i] *= w_;
	return *this;
    }

    vec3 operator/=(float w_) {
	for(int i = 0; i < 3; i++) m_v[i] /= w_;
	return *this;
    }

    vec3 operator+=(const vec3& v) {
	for(int i = 0; i < 3; i++) m_v[i] += v[i];
	return *this;
    }

    vec3 operator-=(const vec3& v) {
	for(int i = 0; i < 3; i++) m_v[i] -= v[i];
	return *this;
    }

    vec3 operator*=(const vec3& v) {
	for(int i = 0; i < 3; i++) m_v[i] *= v[i];
	return *this;
    }

    vec3 operator/=(const vec3& v) {
	for(int i = 0; i < 3; i++) m_v[i] /= v[i];
	return *this;
    }

    float x() const { return m_v[0]; }
    float y() const { return m_v[1]; }
    float z() const { return m_v[2]; }
};

inline vec3 cross(const vec3& v0, const vec3& v1)
{
    vec3 tmp;
    tmp[0] = v0[1] * v1[2] - v0[2] * v1[1];
    tmp[1] = v0[2] * v1[0] - v0[0] * v1[2];
    tmp[2] = v0[0] * v1[1] - v0[1] * v1[0];
    return tmp;
}

struct vec4
{
    float m_v[4];
    static constexpr int vector_size = 4;
    typedef float comp_type;

    vec4(void)
    { }

    void set(float x_, float y_, float z_, float w_)
    {
        m_v[0] = x_; m_v[1] = y_; m_v[2] = z_; m_v[3] = w_;
    }

    vec4(float x_, float y_, float z_, float w_)
    { set(x_, y_, z_, w_); }

    vec4(float v) 
    {
	for(int i = 0; i < 4; i++) m_v[i] = v;
    }

    vec4(const float *v)
    {
	for(int i = 0; i < 4; i++) m_v[i] = v[i];
    }

    vec4(const vec4 &v)
    {
	for(int i = 0; i < 4; i++) m_v[i] = v[i];
    }

    vec4 &operator=(const vec4& v) {
	for(int i = 0; i < 4; i++) m_v[i] = v[i];
	return *this;
    }

    vec4(float *v)
    { set(v); }

    operator const float*() const { return m_v; }
    operator float*() { return m_v; }

    float& operator[] (int i)
        { return m_v[i]; }

    const float& operator[] (int i) const
        { return m_v[i]; }

    void clear() { 
	for(int i = 0; i < 4; i++) m_v[i] = 0;
    }

    void set(const float *v)
	{ for(int i = 0; i < 4; i++) m_v[i] = v[i]; }


    vec4 operator*=(float w_) {
	for(int i = 0; i < 4; i++) m_v[i] *= w_;
	return *this;
    }

    vec4 operator/=(float w_) {
	for(int i = 0; i < 4; i++) m_v[i] /= w_;
	return *this;
    }

    vec4 operator+=(const vec4& v) {
	for(int i = 0; i < 4; i++) m_v[i] += v[i];
	return *this;
    }

    vec4 operator-=(const vec4& v) {
	for(int i = 0; i < 4; i++) m_v[i] -= v[i];
	return *this;
    }

    float x() const { return m_v[0]; }
    float y() const { return m_v[1]; }
    float z() const { return m_v[2]; }
    float w() const { return m_v[3]; }
};

//
// With your left hand up, fingers up, palm facing away, thumb facing to
// the right, thumb is v0-v1, index finger is v0-v2 : plane normal
// sticks out the back of your the hand towards you.
//
inline vec4 make_plane(const vec3& v0, const vec3& v1, const vec3& v2)
{
    vec3 xaxis = v1 - v0;
    vec3 yaxis = v2 - v0;

    vec3 plane = normalize(cross(xaxis, yaxis));

    float D = dot(-v0, plane);
    return vec4(plane[0], plane[1], plane[2], D);
}

struct rot4f : public vec4
{
    rot4f() : vec4() {};
    rot4f(float x_, float y_, float z_, float w_) : vec4(x_, y_, z_, w_) {}

    rot4f(float v) : vec4(v) {}

    rot4f(const float *v) : vec4(v) {}

    rot4f(float *v) : vec4(v) {}

    void set_axis(float x_, float y_, float z_) {
	m_v[1] = x_;
	m_v[2] = y_;
	m_v[3] = z_;
    }

    void set_axis(const vec3 &axis) {
	m_v[1] = axis[0];
	m_v[2] = axis[1];
	m_v[3] = axis[2];
    }

    rot4f& mult(const rot4f& m1, const rot4f &m2);
};

rot4f operator*(const rot4f& r1, const rot4f& r2);

struct mat4f
{
    float m_v[16];
    typedef float comp_type;

    mat4f()
    {
	m_v[0] = 1; m_v[1] = 0; m_v[2] = 0; m_v[3] = 0;
	m_v[4] = 0; m_v[5] = 1; m_v[6] = 0; m_v[7] = 0;
	m_v[8] = 0; m_v[9] = 0; m_v[10] = 1; m_v[11] = 0;
	m_v[12] = 0; m_v[13] = 0; m_v[14] = 0; m_v[15] = 1;
    }

    mat4f(float m00, float m01, float m02, float m03,
	float m10, float m11, float m12, float m13,
	float m20, float m21, float m22, float m23,
	float m30, float m31, float m32, float m33) {

	m_v[0] = m00; m_v[1] = m01; m_v[2] = m02; m_v[3] = m03;
	m_v[4] = m10; m_v[5] = m11; m_v[6] = m12; m_v[7] = m13;
	m_v[8] = m20; m_v[9] = m21; m_v[10] = m22; m_v[11] = m23;
	m_v[12] = m30; m_v[13] = m31; m_v[14] = m32; m_v[15] = m33;
    }

    // mat4f::mult_nm does not perform inverse transpose - just multiplies with
    //     v[3] = 0
    vec3 mult_nm(const vec3 &in) const {
	int i;
	vec4 t;

	for(i = 0; i < 4; i++)
	    t[i] =
		m_v[0 + i] * in[0] + 
		m_v[4 + i] * in[1] + 
		m_v[8 + i] * in[2];

	t[0] /= t[3];
	t[1] /= t[3];
	t[2] /= t[3];
	return vec3(t[0], t[1], t[2]);
    }

    float determinant() const
    {
	return (m_v[0] * m_v[5] - m_v[1] * m_v[4]) *
	    (m_v[10] * m_v[15] - m_v[11] * m_v[14]) + 
	    (m_v[2] * m_v[4] - m_v[0] * m_v[6]) *
	    (m_v[9] * m_v[15] - m_v[11] * m_v[13]) + 
	    (m_v[0] * m_v[7] - m_v[3] * m_v[4]) *
	    (m_v[9] * m_v[14] - m_v[10] * m_v[13]) + 
	    (m_v[1] * m_v[6] - m_v[2] * m_v[5]) *
	    (m_v[8] * m_v[15] - m_v[11] * m_v[12]) + 
	    (m_v[3] * m_v[5] - m_v[1] * m_v[7]) *
	    (m_v[8] * m_v[14] - m_v[10] * m_v[12]) + 
	    (m_v[2] * m_v[7] - m_v[3] * m_v[6]) *
	    (m_v[8] * m_v[13] - m_v[9] * m_v[12]);
    }

    bool invert(const mat4f& in, bool singular_fail = true);
    bool invert() { return invert(*this); }

    static mat4f translation(float x, float y, float z)
    {
	mat4f m;
	m[12] = x;
	m[13] = y;
	m[14] = z;

	return m;
    }

    static mat4f scale(float x, float y, float z)
    {
	mat4f m;
	m[0] = x;
	m[5] = y;
	m[10] = z;

	return m;
    }

    static mat4f rotation(float a, float x, float y, float z)
    {
	mat4f m;
	float c, s, t;

	c = (float)cos(a);
	s = (float)sin(a);
	t = 1.0f - c;

	m[0] = t * x * x + c;
	m[1] = t * x * y + s * z;
	m[2] = t * x * z - s * y;
	m[3] = 0;

	m[4] = t * x * y - s * z;
	m[5] = t * y * y + c;
	m[6] = t * y * z + s * x;
	m[7] = 0;

	m[8] = t * x * z + s * y;
	m[9] = t * y * z - s * x;
	m[10] = t * z * z + c;
	m[11] = 0;

	m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;

	return m;
    }

    static mat4f frustum(float left, float right, float bottom, float top, float nearClip, float farClip)
    {
        mat4f m;

        float A = (right + left) / (right - left);
        float B = (top + bottom) / (top - bottom);
        float C = - (farClip + nearClip) / (farClip - nearClip);
        float D = - 2 * farClip * nearClip / (farClip - nearClip);

        m[0] = 2 * nearClip / (right - left);
        m[5] = 2 * nearClip / (top - bottom);

        m[2] = A;
        m[6] = B;
        m[10] = C;
        m[14] = D;

        m[11] = -1;
        m[15] = 0;

        return m;
    }


    mat4f(const rot4f& r)
    {
	(*this) = rotation(r[0], r[1], r[2], r[3]);
    }

    void calc_rot4f(rot4f *out) const;

    mat4f& mult(mat4f& m1, mat4f &m2)
    {
	mat4f t;
	int i, j;

	for(j = 0; j < 4; j++)
	    for(i = 0; i < 4; i++)
	       t[i * 4 + j] = m1[i * 4 + 0] * m2[0 * 4 + j] +
		   m1[i * 4 + 1] * m2[1 * 4 + j] +
		   m1[i * 4 + 2] * m2[2 * 4 + j] +
		   m1[i * 4 + 3] * m2[3 * 4 + j];

	*this = t;
	return *this;
    }

    mat4f(const float *v)
    {
	for(int i = 0; i < 16; i++) m_v[i] = v[i];
    }

    mat4f(const mat4f &v)
    {
	for(int i = 0; i < 16; i++) m_v[i] = v[i];
    }

    mat4f &operator=(const mat4f& v)
    {
	for(int i = 0; i < 16; i++) m_v[i] = v[i];
	return *this;
    }

    mat4f(float *v)
        { set(v); }

    float& operator[] (int i)
        { return m_v[i]; }

    const float& operator[] (int i) const
        { return m_v[i]; }

    void clear() { 
	for(int i = 0; i < 16; i++) m_v[i] = 0;
    }

    void set(const float *v)
	{ for(int i = 0; i < 16; i++) m_v[i] = v[i]; }

    void store(float *v) const
	{ for(int i = 0; i < 16; i++) v[i] = m_v[i]; }
};

inline mat4f transpose(const mat4f& in)
{
    mat4f t;
    mat4f out;
    int i, j;

    t = in;
    for(i = 0; i < 4; i++) {
        for(j = 0; j < 4; j++)  {
            out.m_v[i + j * 4] = t[j + i * 4];
        }
    }

    return out;
}

constexpr float MAT4F_EPSILON = .00001f;

inline mat4f inverse(const mat4f& mat)
{
    int		i, rswap;
    [[maybe_unused]] float	det;
    float       div, swap;
    mat4f	hold;
    mat4f       result;

    hold = mat;

#if 0
    det = mat.determinant();
    if(fabsf(det) < MAT4F_EPSILON) {
        /* singular? */
	return mat4f();
    }
#endif

    rswap = 0;
    /* this loop isn't entered unless [0 + 0] > MAT4F_EPSILON and det > MAT4F_EPSILON,
	 so rswap wouldn't be 0, but I initialize so as not to get warned */
    if(fabs(hold[0]) < MAT4F_EPSILON)
    {
        if(fabs(hold[1]) > MAT4F_EPSILON)
            rswap = 1;
        else if(fabs(hold[2]) > MAT4F_EPSILON)
	    rswap = 2;
        else if(fabs(hold[3]) > MAT4F_EPSILON)
	    rswap = 3;

        for(i = 0; i < 4; i++)
	{
            swap = hold[i * 4 + 0];
            hold[i * 4 + 0] = hold[i * 4 + rswap];
            hold[i * 4 + rswap] = swap;

            swap = result.m_v[i * 4 + 0];
            result.m_v[i * 4 + 0] = result.m_v[i * 4 + rswap];
            result.m_v[i * 4 + rswap] = swap;
        }
    }
        
    div = hold[0];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 0] /= div;
        result.m_v[i * 4 + 0] /= div;
    }

    div = hold[1];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 1] -= div * hold[i * 4 + 0];
        result.m_v[i * 4 + 1] -= div * result.m_v[i * 4 + 0];
    }
    div = hold[2];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 2] -= div * hold[i * 4 + 0];
        result.m_v[i * 4 + 2] -= div * result.m_v[i * 4 + 0];
    }
    div = hold[3];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 3] -= div * hold[i * 4 + 0];
        result.m_v[i * 4 + 3] -= div * result.m_v[i * 4 + 0];
    }

    if(fabs(hold[5]) < MAT4F_EPSILON){
        if(fabs(hold[6]) > MAT4F_EPSILON)
	    rswap = 2;
        else if(fabs(hold[7]) > MAT4F_EPSILON)
	    rswap = 3;

        for(i = 0; i < 4; i++)
	{
            swap = hold[i * 4 + 1];
            hold[i * 4 + 1] = hold[i * 4 + rswap];
            hold[i * 4 + rswap] = swap;

            swap = result.m_v[i * 4 + 1];
            result.m_v[i * 4 + 1] = result.m_v[i * 4 + rswap];
            result.m_v[i * 4 + rswap] = swap;
        }
    }

    div = hold[5];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 1] /= div;
        result.m_v[i * 4 + 1] /= div;
    }

    div = hold[4];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 0] -= div * hold[i * 4 + 1];
        result.m_v[i * 4 + 0] -= div * result.m_v[i * 4 + 1];
    }
    div = hold[6];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 2] -= div * hold[i * 4 + 1];
        result.m_v[i * 4 + 2] -= div * result.m_v[i * 4 + 1];
    }
    div = hold[7];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 3] -= div * hold[i * 4 + 1];
        result.m_v[i * 4 + 3] -= div * result.m_v[i * 4 + 1];
    }

    if(fabs(hold[10]) < MAT4F_EPSILON){
        for(i = 0; i < 4; i++)
	{
            swap = hold[i * 4 + 2];
            hold[i * 4 + 2] = hold[i * 4 + 3];
            hold[i * 4 + 3] = swap;

            swap = result.m_v[i * 4 + 2];
            result.m_v[i * 4 + 2] = result.m_v[i * 4 + 3];
            result.m_v[i * 4 + 3] = swap;
        }
    }

    div = hold[10];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 2] /= div;
        result.m_v[i * 4 + 2] /= div;
    }

    div = hold[8];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 0] -= div * hold[i * 4 + 2];
        result.m_v[i * 4 + 0] -= div * result.m_v[i * 4 + 2];
    }
    div = hold[9];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 1] -= div * hold[i * 4 + 2];
        result.m_v[i * 4 + 1] -= div * result.m_v[i * 4 + 2];
    }
    div = hold[11];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 3] -= div * hold[i * 4 + 2];
        result.m_v[i * 4 + 3] -= div * result.m_v[i * 4 + 2];
    }

    div = hold[15];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 3] /= div;
        result.m_v[i * 4 + 3] /= div;
    }

    div = hold[12];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 0] -= div * hold[i * 4 + 3];
        result.m_v[i * 4 + 0] -= div * result.m_v[i * 4 + 3];
    }
    div = hold[13];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 1] -= div * hold[i * 4 + 3];
        result.m_v[i * 4 + 1] -= div * result.m_v[i * 4 + 3];
    }
    div = hold[14];
    for(i = 0; i < 4; i++)
    {
        hold[i * 4 + 2] -= div * hold[i * 4 + 3];
        result.m_v[i * 4 + 2] -= div * result.m_v[i * 4 + 3];
    }
    
    return result;
}


inline mat4f operator*(const mat4f& m1, const mat4f& m2)
{
    mat4f t;
    int i, j;

    for(j = 0; j < 4; j++)
	for(i = 0; i < 4; i++)
	   t[i * 4 + j] =
	       m1[i * 4 + 0] * m2[0 * 4 + j] +
	       m1[i * 4 + 1] * m2[1 * 4 + j] +
	       m1[i * 4 + 2] * m2[2 * 4 + j] +
	       m1[i * 4 + 3] * m2[3 * 4 + j];

    return t;
}

inline vec4 operator*(const vec4& in, const mat4f& m)
{
    int i;
    vec4 t;

    for(i = 0; i < 4; i++)
	t[i] =
	    m[0 + i] * in[0] + 
	    m[4 + i] * in[1] + 
	    m[8 + i] * in[2] + 
	    m[12 + i] * in[3];
    return t;
}

#if 0
inline vec3 operator*(const vec3& in, const mat4f& m)
{
    int i;
    vec3 t;

    for(i = 0; i < 3; i++)
	t[i] =
	    m[0 + i] * in[0] + 
	    m[4 + i] * in[1] + 
	    m[8 + i] * in[2] + 
	    m[12 + i];
    return t;
}
#else
inline vec3 operator*(const vec3& in, const mat4f& m)
{
    int i;
    vec4 t;

    for(i = 0; i < 4; i++)
	t[i] =
	    m[0 + i] * in[0] + 
	    m[4 + i] * in[1] + 
	    m[8 + i] * in[2] + 
	    m[12 + i];
    return vec3(t.m_v);
}
#endif

enum axis_t { X_AXIS = 0, Y_AXIS = 1, Z_AXIS = 2};

struct segment
{
    vec3 m_v0;
    vec3 m_v1;
    segment(const vec3 &v0, const vec3 &v1) :
        m_v0(v0),
        m_v1(v1)
    {}
};

struct ray
{
    vec3 o;
    vec3 d;

    ray(const vec3& o, const vec3& d) : o(o), d(d)
    { }

    ray(const segment &s) :
        o(s.m_v0),
        d(s.m_v1 - s.m_v0)
    { }

    ray() {}

    float length() const
    {
        return ::length(d);
    }

    float at(int axis, float plane) const
    {
	if(d[axis] > -.00001f && d[axis] < 0.0f)
	    return -std::numeric_limits<float>::max();
	if(d[axis] >= 0.0f && d[axis] < 0.00001f)
	    return std::numeric_limits<float>::max();
	return (plane - o[axis]) / d[axis];
    }

    vec3 at(float t) const
    {
        return o + d * t;
    }
};

// Probably should do this by passing in inverse-transpose for direction
inline ray operator*(const ray& r, const mat4f& m)
{
    vec3 newo = r.o * m;
    vec3 newd = (r.d + r.o) * m - newo;
    return ray(newo, newd);
}

// Distance along r to plane
inline float operator*(const ray& r, const vec4& plane)
{
    vec3 normal(plane[0], plane[1], plane[2]);

    float factor = dot(r.d, normal);
    return (plane[3] - dot(normal, r.o)) / factor;
}

inline void transform_ray(const vec3& origin, const vec3& direction, const mat4f& m, vec3 *neworigin, vec3 *newdirection)
{
    vec3 oldorigin = origin; // in case in and out are the same

    *neworigin = oldorigin * m;
    *newdirection = (direction + oldorigin) * m - *neworigin;
}

inline void transform_ray(const vec4& origin, const vec4& direction, const mat4f& m, vec4 *neworigin, vec4 *newdirection)
{
    vec4 oldorigin = origin; // in case in and out are the same

    *neworigin = oldorigin * m;
    *newdirection = (oldorigin + direction) * m - *neworigin;
}

// XXX Should be able to use normal matrix to transform ray direction...
inline void transform_ray(vec3* origin, vec3* direction, const mat4f& m)
{
    vec3 neworigin, newdirection;

    neworigin = *origin * m;
    newdirection = (*direction + *origin) * m - neworigin;

    *origin = neworigin;
    *direction = newdirection;
}

struct aabox
{
    vec3 boxmin;
    vec3 boxmax;
    aabox(const vec3& boxmin_, const vec3& boxmax_) :
        boxmin(boxmin_),
        boxmax(boxmax_)
    {}
    aabox() :
        boxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()),
        boxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max())
    {
    }
    void clear()
    {
        boxmin = vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        boxmax = vec3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    }
    vec3 center() const
    {
        return (boxmin + boxmax) * .5;
    }
    vec3 dim() const // for any dimension for which min > max, returns 0
    {
        vec3 d = boxmax - boxmin;
        return vec3(std::max(d[0], 0.0f), std::max(d[1], 0.0f), std::max(d[2], 0.0f));
    }
    aabox& add(const vec3& v)
    {
        const float bumpout = .00001f;
        boxmin = min(boxmin, v - bumpout);
        boxmax = max(boxmax, v + bumpout);
        return *this;
    }
    aabox& operator+=(const vec3& v)
    {
        return add(v);
    }
    aabox& add(const vec3& c, float r)
    {
        const float bumpout = 1.0001f;
        boxmin = min(boxmin, c - r * bumpout);
        boxmax = max(boxmax, c + r * bumpout);
        return *this;
    }
    aabox& add(const vec3& addmin, const vec3& addmax)
    {
        boxmin = min(addmin, boxmin);
        boxmax = max(addmax, boxmax);
        return *this;
    }
    aabox& add(const aabox& box)
    {
        boxmin = min(box.boxmin, boxmin);
        boxmax = max(box.boxmax, boxmax);
        return *this;
    }
    aabox& operator+=(const aabox& box)
    {
        return add(box);
    }
    aabox& add(const vec3& v0, const vec3& v1, const vec3& v2)
    {
        add(v0);
        add(v1);
        add(v2);
        return *this;
    }
    vec3 diagonal() const
    {
        return boxmax - boxmin;
    }
    aabox operator*(const mat4f& m) const
    {
        aabox newbox;
        newbox.add(vec3(boxmin[0], boxmin[1], boxmin[2]) * m);
        newbox.add(vec3(boxmax[0], boxmin[1], boxmin[2]) * m);
        newbox.add(vec3(boxmin[0], boxmax[1], boxmin[2]) * m);
        newbox.add(vec3(boxmax[0], boxmax[1], boxmin[2]) * m);
        newbox.add(vec3(boxmin[0], boxmin[1], boxmax[2]) * m);
        newbox.add(vec3(boxmax[0], boxmin[1], boxmax[2]) * m);
        newbox.add(vec3(boxmin[0], boxmax[1], boxmax[2]) * m);
        newbox.add(vec3(boxmax[0], boxmax[1], boxmax[2]) * m);
        return newbox;
    }
};

inline void add_sphere(vec3 *C1, float *R1, const vec3& C2, float R2)
{
    static float epsilon = .000001f;

    vec3 d = C2 - *C1;
    float len = sqrtf(dot(d, d));

    if(len + R2 <= *R1) {
        *R1 += epsilon;
        return;
    }

    if(len + *R1 <= R2) {
        *C1 = C2;
        *R1 = R2 + epsilon;
        return;
    }

    vec3 dhat = d - len;
    float rprime = (*R1 + R2 + len) / 2;
    vec3 cprime = *C1 + dhat * (rprime - *R1);

    *C1 = cprime;
    *R1 = rprime + epsilon;
}

struct range
{
    float t0, t1;
    range() :
        t0(-std::numeric_limits<float>::max()),
        t1(std::numeric_limits<float>::max())
    {}
#if 0
    range(const range &r) :
        t0(r.t0),
        t1(r.t1)
    {}
#endif
    range(float t0_, float t1_) :
        t0(t0_),
        t1(t1_)
    {}
    operator bool() { return t0 < t1; }
    range &intersect(const range &r2)
    {
        t0 = std::max(t0, r2.t0);
        t1 = std::min(t1, r2.t1);
        return *this;
    }
};

inline range ray_intersect_box(const aabox& box, const ray& theray)
{
    float t0, t1;

    t0 = (box.boxmin.x() - theray.o.x()) / theray.d.x();
    t1 = (box.boxmax.x() - theray.o.x()) / theray.d.x();
    range r = (theray.d.x() >= 0.0) ? range(t0, t1) : range(t1, t0);

    t0 = (box.boxmin.y() - theray.o.y()) / theray.d.y();
    t1 = (box.boxmax.y() - theray.o.y()) / theray.d.y();
    if(theray.d.y() >= 0.0) {
        r.intersect(range(t0, t1));
    } else {
        r.intersect(range(t1, t0));
    }

    t0 = (box.boxmin.z() - theray.o.z()) / theray.d.z();
    t1 = (box.boxmax.z() - theray.o.z()) / theray.d.z();
    if(theray.d.z() >= 0.0) {
        r.intersect(range(t0, t1));
    } else {
        r.intersect(range(t1, t0));
    }

    return r;
}


#endif /* __VECTORMATH_H__ */
