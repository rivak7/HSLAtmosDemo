import numpy as np
import taichi as ti
from pathlib import Path
vec3 = ti.math.vec3
dot = ti.Vector.dot

#region CONSTANTS
# feel free to change these (reasonable precision with citations might get you points!)
EARTH_RADIUS_KM = 6000.0 # if you modify this make sure to update line 38 of demo.py
ATMOSPHERE_THICKNESS_KM = 110 # The true atmosphere extends quite far, so we choose a point where it visually seems to end
# ^ this atmosphere is actually very innacurate, but sometimes it's instructive to have it be big to see what's going on
# be careful though, sometimes having a thick atmosphere leads to unintended consequences
MOLECULAR_SCALE_HEIGHT_KM = 8.0 # H
#endregion

#region HELPER FUNCTIONS

# This function models the density of the atmosphere with altitude using p(h) = p_0 * e^(-h/H)
@ti.func
def atmospheric_density(pos: vec3) -> ti.f32:
    altitude_km = pos.norm() - EARTH_RADIUS_KM # altitude above surface
    return ti.exp(-altitude_km / MOLECULAR_SCALE_HEIGHT_KM) if (altitude_km >= 0.0 and altitude_km <= ATMOSPHERE_THICKNESS_KM) else 0

NUM_VIEW_SAMPLES = 16 # use 16 samples for numerical integration
@ti.func
def integrate_density(start: vec3, end: vec3) -> ti.f32:
    segment = end - start
    segment_length = segment.norm()
    step = segment / NUM_VIEW_SAMPLES
    step_length = segment_length / NUM_VIEW_SAMPLES

    total_density = 0.0

    for i in ti.static(range(NUM_VIEW_SAMPLES)):
        sample_pos = start + step * (ti.cast(i, ti.f32) + 0.5)
        total_density += atmospheric_density(sample_pos) * step_length

    return total_density

# This function transforms the ray into a coordinate space where the z axis is scaled by width/height
@ti.func
def _convert_ray_to_sphere_space(origin: vec3, direction: vec3, width: ti.f32, height: ti.f32):
    w_over_h = width / height
    sphere_origin = vec3(origin.x, origin.y, origin.z * w_over_h)
    sphere_dir = vec3(direction.x, direction.y, direction.z * w_over_h)
    return sphere_origin, sphere_dir.normalized()

# This function transforms a point into a coordinate space where the z axis is scaled by height/width
@ti.func
def _convert_pos_to_spheroid_space(pos: vec3, width: ti.f32, height: ti.f32) -> vec3:
    return vec3(pos.x, pos.y, pos.z / (width / height))


# Given an origin and a ray, this function returns up to two points of intersection,
# along with booleans for whether the collision point exists
#
# I wonder why this function generalizes to spheroids?
@ti.func
def cast_ray_against_oblate_spheroid(origin: vec3, direction: vec3, width: ti.f32, height: ti.f32):
    sphere_origin, sphere_dir = _convert_ray_to_sphere_space(origin, direction, width, height)

    a = sphere_dir.dot(sphere_dir)
    b = 2.0 * sphere_origin.dot(sphere_dir)
    c = sphere_origin.dot(sphere_origin) - width * width
    determinant = (-4.0 * c * a) + (b * b)

    first_pos = vec3(9.0, 9.0, 9.0)
    collides_first = False
    second_pos = vec3(0.0, 0.0, 0.0)
    collides_second = False

    if determinant >= 0.0:
        sqrt_det = ti.sqrt(determinant)
        two_a = 2.0 * a
        small_t = (-b - sqrt_det) / two_a
        large_t = (-b + sqrt_det) / two_a

        first_pos = vec3(0.0, 0.0, 0.0)
        second_pos = vec3(0.0, 0.0, 0.0)

        if small_t >= 0.0:
            hit = sphere_origin + sphere_dir * small_t
            first_pos = _convert_pos_to_spheroid_space(hit, width, height)
            collides_first = True
        if large_t >= 0.0:
            hit = sphere_origin + sphere_dir * large_t
            second_pos = _convert_pos_to_spheroid_space(hit, width, height)
            collides_second = True

    return first_pos, collides_first, second_pos, collides_second
#endregion

# notice every function needs @ti.func so it can run in the kernel
@ti.func
def funnyFunction(pos, ray, sun_dir):
    atmosphereHitPosition, collided,  _, _ = cast_ray_against_oblate_spheroid(pos, ray, EARTH_RADIUS_KM+ATMOSPHERE_THICKNESS_KM, EARTH_RADIUS_KM+ATMOSPHERE_THICKNESS_KM)
    funGradient = vec3(0,0,0)
    if (collided):
        hit_dir = atmosphereHitPosition.normalized()
        # notice you can sometimes get weird behavior when there's negative values
        color_term = (hit_dir*0.5+vec3(0.5,0.5,0.8)) * max(dot(hit_dir, sun_dir),0.05)
        earth_term = _earth(pos, ray, sun_dir)*vec3(0.5,0.5,0.7) #earth coloring
        haze = vec3(0.05,0.02,0.05) #haze due to uniform atmosphere
        funGradient = color_term + earth_term + haze
    else:
        funGradient = (ray-vec3(0.5,0,0))*0.5+vec3(0.5,0.5,0.5)
    return funGradient

@ti.func
def funnyFunction(pos, ray, sun_dir):
    first_pos, first_hit, second_pos, second_hit = \
        cast_ray_against_oblate_spheroid(
            pos,
            ray,
            EARTH_RADIUS_KM + ATMOSPHERE_THICKNESS_KM,
            EARTH_RADIUS_KM + ATMOSPHERE_THICKNESS_KM,
        )
    color = vec3(0.0, 0.0, 0.0)

    if first_hit and second_hit:
        density = integrate_density(first_pos, second_pos)
        # Debug visualization only: compress column density into a 0-1 range
        debug_value = 1.0 - ti.exp(-0.01 * density)
        color = vec3(debug_value, debug_value, debug_value)

    return color

# the main function that demo.py calls to get the color of the atmosphere at a certain pixel
@ti.func
def _atmos(pos, ray, sun_dir):

    funGradient = funnyFunction(pos, ray, sun_dir)
    black = vec3(0,0,0)

    # return the color, then a debug color
    return black, funGradient


# Used by demo.py
# This earth function is very simple, it renders a blueish-grey ball
# Do not touch if you are competing for accuracy, unless you are changing earth's radius
# A simple earth helps the judges see your atmosphere more clearly
#
# Feel free to play with it if you're making an art piece
@ti.func
def _earth(pos, direction, sun_dir):
    surface_km, hit, _, _ = cast_ray_against_oblate_spheroid(pos, direction, EARTH_RADIUS_KM, EARTH_RADIUS_KM)
    color = vec3(0,0,0)
    if hit:
        surface_dir = surface_km.normalized()
        ndotl = max(surface_dir.dot(sun_dir),0)
        if ndotl >= 0.0:
            color = vec3(0.7,0.7,1)*ndotl*0.5
    return color
