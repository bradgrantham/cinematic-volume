//
// Copyright 2013-2014, Bradley A. Grantham
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//      http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 

#include "manipulator.h"

/* Win32 math.h doesn't define M_PI. */
#if defined(_WIN32)
#if !defined(M_PI)
#define M_PI 3.14159265
#endif /* !defined(M_PI) */
#endif /* defined(WIN32) */

/* internal use */

static void drag_to_rotation(float dx, float dy, rot4f *rotation)
{
    float dist;

    /* XXX grantham 990825 - this "dist" doesn't make me confident. */
    /* but I put in the *10000 to decrease chance of underflow  (???) */
    dist = sqrt(dx * 10000 * dx * 10000 + dy * 10000 * dy * 10000) / 10000;
    /* dist = sqrt(dx * dx + dy * dy); */

    rotation->set(3.14159265358979323846f * dist, dy / dist, dx / dist, 0.0f);
}

static void calc_view_matrix(rot4f view_rotation, vec3 view_offset,
    vec3 obj_center, vec3 obj_scale, mat4f *view_matrix)
{
    *view_matrix = mat4f::translation(view_offset[0], view_offset[1], view_offset[2]);
    *view_matrix = mat4f(view_rotation) * *view_matrix;

    *view_matrix = mat4f::scale(obj_scale[0], obj_scale[1], obj_scale[2]) * *view_matrix;
    *view_matrix = mat4f::translation(-obj_center[0], -obj_center[1], -obj_center[2]) * *view_matrix;
}

/* external API */

void manipulator::calculate_matrix()
{
    calc_view_matrix(m_rotation, m_translation, m_center,
        m_scale, &m_matrix);
}

void manipulator::set_frame(const mat4f& frame)
{
    mat4f i;
    vec3 origin;

    m_frame = frame;

    i.invert(m_frame);
    origin = vec3(0.0f, 0.0f, 0.0f) * i;
    m_worldX = vec3(1.0f, 0.0f, 0.0f) * i - origin;
    m_worldY = vec3(0.0f, 1.0f, 0.0f) * i - origin;
    m_worldZ = vec3(0.0f, 0.0f, 1.0f) * i - origin;
}

void manipulator::move(float dx, float dy)
{
    switch(m_mode) {

	case ROTATE:
	    if(dx != 0 || dy != 0) {
		rot4f local_rotation;
		rot4f world_rotation;
		drag_to_rotation(dx, dy, &world_rotation);
		local_rotation[0] = world_rotation[0];
		vec3 axis = m_worldX * world_rotation[1] + 
                    m_worldY * world_rotation[2];
		local_rotation.set_axis(axis);
		m_rotation = m_rotation * local_rotation;
	    }
	    break;

	case ROLL: {
            float angle = 3.14159265358979323846f * 2.0f * -dy;
            m_rotation = m_rotation * rot4f(angle, 0.0f, 0.0f, 1.0f);
	    break;
        }

	case SCROLL:
	    m_translation = m_translation +
                m_worldX * dx * m_reference_size * m_motion_scale +
                m_worldY * -dy * m_reference_size * m_motion_scale;
	    break;

	case DOLLY:
	    m_translation = m_translation + m_worldZ * dy * m_reference_size * m_motion_scale;
	    break;
    }

    calc_view_matrix(m_rotation, m_translation, m_center,
        m_scale, &m_matrix);
}

manipulator::manipulator()
{
    m_worldX = vec3(1.0f, 0.0f, 0.0f);
    m_worldY = vec3(0.0f, 1.0f, 0.0f);
    m_worldZ = vec3(0.0f, 0.0f, 1.0f);

    m_center.set(0, 0, 0);

    m_scale.set(1, 1, 1);

    /* diagonal of 2-high cube */
    m_reference_size = 3.465f;

    m_motion_scale = 1.0;

    m_translation.set(0, 0, 0);

    m_rotation.set(0, 1, 0, 0);

    calc_view_matrix(m_rotation, m_translation, m_center,
        m_scale, &m_matrix);

    m_mode = ROTATE;
}

manipulator::manipulator(const aabox& bounds, float fov)
{
    m_worldX = vec3(1.0f, 0.0f, 0.0f);
    m_worldY = vec3(0.0f, 1.0f, 0.0f);
    m_worldZ = vec3(0.0f, 0.0f, 1.0f);

    m_center = bounds.center();

    m_scale = vec3(1.0f, 1.0f, 1.0f);

    m_reference_size = length(bounds.dim());
    m_motion_scale = 1.0;

    m_translation.set(0, 0, -m_reference_size / cosf(fov));

    m_rotation.set(0, 1, 0, 0);

    calc_view_matrix(m_rotation, m_translation, m_center,
        m_scale, &m_matrix);

    m_mode = manipulator::ROTATE;
}

