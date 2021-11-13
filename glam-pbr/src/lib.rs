#![no_std]

use core::f32::consts::FRAC_1_PI;
use glam::Vec3;
use num_traits::Float;

// Workarounds: can't use f32.lerp, f32.clamp or f32.powi.

#[derive(Copy, Clone)]
pub struct View(pub Vec3);

impl ShadingVector for View {
    fn vector(&self) -> Vec3 {
        self.0
    }
}

#[derive(Copy, Clone)]
pub struct Light(pub Vec3);

impl ShadingVector for Light {
    fn vector(&self) -> Vec3 {
        self.0
    }
}

/// A vector used in shading. It is important that the vector is normalised and points away from the surface of the object being shaded.
pub trait ShadingVector {
    fn vector(&self) -> Vec3;
}

#[derive(Copy, Clone)]
pub struct Normal(pub Vec3);

impl ShadingVector for Normal {
    fn vector(&self) -> Vec3 {
        self.0
    }
}

#[derive(Copy, Clone)]
pub struct Halfway(Vec3);

impl Halfway {
    pub fn new(view: &View, light: &Light) -> Self {
        Self((view.0 + light.0).normalize())
    }
}

impl ShadingVector for Halfway {
    fn vector(&self) -> Vec3 {
        self.0
    }
}

pub struct Dot<A, B> {
    pub value: f32,
    _phantom: core::marker::PhantomData<(A, B)>,
}

impl<A, B> Clone for Dot<A, B> {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<A, B> Copy for Dot<A, B> {}

impl<A: ShadingVector, B: ShadingVector> Dot<A, B> {
    pub fn new(a: &A, b: &B, epsilon: f32) -> Self {
        fn clamp(value: f32, min: f32, max: f32) -> f32 {
            value.max(min).min(max)
        }

        Self {
            value: clamp(a.vector().dot(b.vector()), epsilon, 1.0),
            _phantom: core::marker::PhantomData,
        }
    }
}

pub fn d_ggx(normal_dot_halfway: Dot<Normal, Halfway>, roughness: ActualRoughness) -> f32 {
    let noh = normal_dot_halfway.value;

    let a = noh * roughness.0;
    let k = roughness.0 / (1.0 - noh * noh + a * a);

    k * k * FRAC_1_PI
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
// Geometric shadowing function

pub fn v_smith_ggx_correlated(
    normal_dot_view: Dot<Normal, View>,
    normal_dot_light: Dot<Normal, Light>,
    roughness: ActualRoughness,
) -> f32 {
    let nov = normal_dot_view.value;
    let nol = normal_dot_light.value;

    let a2 = roughness.0 * roughness.0;
    let ggx_v = nol * (nov * nov * (1.0 - a2) + a2).sqrt();
    let ggx_l = nov * (nol * nol * (1.0 - a2) + a2).sqrt();
    0.5 / (ggx_v + ggx_l)
}

// Fresnel

pub fn fresnel_schlick(light_dot_halfway: Dot<Light, Halfway>, f0: Vec3, f90: Vec3) -> Vec3 {
    f0 + (f90 - f0) * (1.0 - light_dot_halfway.value).powf(5.0)
}

// Diffuse

pub const fn fd_lambert() -> f32 {
    FRAC_1_PI
}

// Disney diffuse (more fun!)

/// Compute f90 according to burley.
pub fn compute_f90(light_dot_halfway: Dot<Light, Halfway>, roughness: ActualRoughness) -> f32 {
    let loh = light_dot_halfway.value;

    0.5 + 2.0 * roughness.0 * loh * loh
}

pub fn fd_burley(
    normal_dot_view: Dot<Normal, View>,
    normal_dot_light: Dot<Normal, Light>,
    light_dot_halfway: Dot<Light, Halfway>,
    roughness: ActualRoughness,
) -> f32 {
    // Internal untyped fresnel function for burley.
    fn fresnel_schlick(u: f32, f0: f32, f90: f32) -> f32 {
        f0 + (f90 - f0) * (1.0 - u).powf(5.0)
    }

    let nov = normal_dot_view.value;
    let nol = normal_dot_light.value;

    let f90 = compute_f90(light_dot_halfway, roughness);
    let light_scatter = fresnel_schlick(nol, 1.0, f90);
    let view_scatter = fresnel_schlick(nov, 1.0, f90);
    light_scatter * view_scatter * FRAC_1_PI
}

#[derive(Copy, Clone)]
pub struct ActualRoughness(f32);

#[derive(Clone, Copy)]
pub struct PerceptualRoughness(pub f32);

impl PerceptualRoughness {
    pub fn as_actual_roughness(&self) -> ActualRoughness {
        ActualRoughness(self.0 * self.0)
    }
}

pub struct BasicBrdfParams {
    pub normal: Normal,
    pub light: Light,
    pub light_intensity: Vec3,
    pub view: View,
    pub material_params: MaterialParams,
}

#[derive(Clone, Copy)]
pub struct MaterialParams {
    pub diffuse_colour: Vec3,
    pub metallic: f32,
    pub perceptual_roughness: PerceptualRoughness,
    pub perceptual_dielectric_reflectance: PerceptualDielectricReflectance,
}

#[derive(Clone, Copy)]
pub struct PerceptualDielectricReflectance(pub f32);

/// Corresponds a f0 of 4% reflectance on non-metallic (dielectric) materials (0.16 * 0.5 * 0.5).
impl Default for PerceptualDielectricReflectance {
    fn default() -> Self {
        Self(0.5)
    }
}

impl PerceptualDielectricReflectance {
    pub fn to_dielectric_f0(&self) -> f32 {
        0.16 * self.0 * self.0
    }
}

pub fn basic_brdf(params: BasicBrdfParams) -> Vec3 {
    let BasicBrdfParams {
        normal,
        light,
        light_intensity,
        view,
        material_params:
            MaterialParams {
                diffuse_colour,
                metallic,
                perceptual_roughness,
                perceptual_dielectric_reflectance,
            },
    } = params;

    let actual_roughness = perceptual_roughness.as_actual_roughness();

    let halfway = Halfway::new(&view, &light);
    let normal_dot_halfway = Dot::new(&normal, &halfway, 0.0);
    let light_dot_halfway = Dot::new(&light, &halfway, 0.0);
    let normal_dot_view = Dot::new(&normal, &view, 0.0);
    let normal_dot_light = Dot::new(&normal, &light, 0.0);

    let f0 = compute_f0(metallic, perceptual_dielectric_reflectance, diffuse_colour);
    let f90 = compute_f90(light_dot_halfway, actual_roughness);
    let fresnel = fresnel_schlick(light_dot_halfway, f0, Vec3::splat(f90));

    let distribution_function = d_ggx(normal_dot_halfway, actual_roughness);

    let geometric_shadowing =
        v_smith_ggx_correlated(normal_dot_view, normal_dot_light, actual_roughness);

    // Specular BRDF factor.
    let specular_brdf_factor = (distribution_function * geometric_shadowing) * fresnel;

    // Diffuse BRDF factor.
    let diffuse_brdf_factor = diffuse_colour
        * fd_burley(
            normal_dot_view,
            normal_dot_light,
            light_dot_halfway,
            actual_roughness,
        );

    let combined_factor = diffuse_brdf_factor + specular_brdf_factor;

    light_intensity * normal_dot_light.value * combined_factor
}

pub fn compute_f0(
    metallic: f32,
    perceptual_dielectric_reflectance: PerceptualDielectricReflectance,
    diffuse_colour: Vec3,
) -> Vec3 {
    // from:
    // https://google.github.io/filament/Filament.md.html#materialsystem/parameterization/remapping
    let dielectric_f0 = perceptual_dielectric_reflectance.to_dielectric_f0();
    let metallic_f0 = diffuse_colour;

    (1.0 - metallic) * dielectric_f0 + metallic * metallic_f0
}
