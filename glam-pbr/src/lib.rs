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
    pub fn new(a: &A, b: &B) -> Self {
        Self {
            value: a.vector().dot(b.vector()).max(core::f32::EPSILON),
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

pub fn fresnel_schlick(view_dot_halfway: Dot<View, Halfway>, f0: Vec3, f90: Vec3) -> Vec3 {
    f0 + (f90 - f0) * (1.0 - view_dot_halfway.value).powf(5.0)
}

#[derive(Copy, Clone)]
pub struct ActualRoughness(f32);

impl ActualRoughness {
    fn apply_ior(self, ior: IndexOfRefraction) -> ActualRoughness {
        ActualRoughness(self.0 * clamp(ior.0 * 2.0 - 2.0, 0.0, 1.0))
    }
}

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
    pub index_of_refraction: IndexOfRefraction,
}

#[derive(Clone, Copy)]
pub struct IndexOfRefraction(pub f32);

/// Corresponds a f0 of 4% reflectance on dielectrics ((1.0 - ior) / (1.0 + ior)) ^ 2.
impl Default for IndexOfRefraction {
    fn default() -> Self {
        Self(1.5)
    }
}

impl IndexOfRefraction {
    pub fn to_dielectric_f0(&self) -> f32 {
        let root = (1.0 - self.0) / (1.0 + self.0);
        root * root
    }
}

/*
fn specular_btdf(
    actual_roughness: ActualRoughness,
    index_of_refraction: IndexOfRefraction,
    normal: Normal,
    view: View,
    light: Light,
    base: Vec3,
    f0: Vec3,
    f90: Vec3,
) -> Vec3 {
    let transmission_roughness = actual_roughness.apply_ior(index_of_refraction);

    let light_mirrored = Light((light.0 * 2.0 * normal.0 * (-light.0).dot(normal.0)).normalize());

    let halfway = Halfway::new(&view, &light_mirrored);
    let normal_dot_halfway = Dot::new(&normal, &halfway);
    let view_dot_halfway = Dot::new(&view, &halfway);
    let normal_dot_view = Dot::new(&normal, &view);
    let normal_dot_light_mirrored = Dot::new(&normal, &light_mirrored);

    let distribution = d_ggx(normal_dot_halfway, transmission_roughness);

    let geometric_shadowing =
        v_smith_ggx_correlated(normal_dot_view, normal_dot_light_mirrored, transmission_roughness);

    let fresnel = fresnel_schlick(view_dot_halfway, f0, f90);

    (1.0 - fresnel) * base * distribution * geometric_shadowing
}
*/

fn diffuse_brdf(base: Vec3, fresnel: Vec3) -> Vec3 {
    (1.0 - fresnel) * FRAC_1_PI * base
}

fn specular_brdf(
    normal_dot_view: Dot<Normal, View>,
    normal_dot_light: Dot<Normal, Light>,
    normal_dot_halfway: Dot<Normal, Halfway>,
    actual_roughness: ActualRoughness,
    fresnel: Vec3,
) -> Vec3 {
    let distribution_function = d_ggx(normal_dot_halfway, actual_roughness);

    let geometric_shadowing =
        v_smith_ggx_correlated(normal_dot_view, normal_dot_light, actual_roughness);

    (distribution_function * geometric_shadowing) * fresnel
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
                index_of_refraction,
            },
    } = params;

    let actual_roughness = perceptual_roughness.as_actual_roughness();

    let halfway = Halfway::new(&view, &light);
    let normal_dot_halfway = Dot::new(&normal, &halfway);
    let normal_dot_view = Dot::new(&normal, &view);
    let normal_dot_light = Dot::new(&normal, &light);
    let view_dot_halfway = Dot::new(&view, &halfway);

    let c_diff = diffuse_colour.lerp(Vec3::ZERO, metallic);
    let f0 = {
        Vec3::splat(index_of_refraction.to_dielectric_f0()).lerp(diffuse_colour, metallic)
    };

    let fresnel = fresnel_schlick(view_dot_halfway, f0, Vec3::splat(1.0));

    let material = diffuse_brdf(c_diff, fresnel) + specular_brdf(normal_dot_view, normal_dot_light, normal_dot_halfway, actual_roughness, fresnel);

    light_intensity * normal_dot_light.value * material
}

pub fn compute_f0(
    metallic: f32,
    index_of_refraction: IndexOfRefraction,
    diffuse_colour: Vec3,
) -> Vec3 {
    // from:
    // https://google.github.io/filament/Filament.md.html#materialsystem/parameterization/remapping
    let dielectric_f0 = index_of_refraction.to_dielectric_f0();
    let metallic_f0 = diffuse_colour;

    (1.0 - metallic) * dielectric_f0 + metallic * metallic_f0
}

#[test]
fn test_i_havent_broken_anything() {
    let params = BasicBrdfParams {
        normal: Normal(Vec3::Y),
        light: Light(Vec3::new(1.0, 1.0, 0.0).normalize()),
        light_intensity: Vec3::ONE,
        view: View(Vec3::new(0.0, 1.0, 1.0).normalize()),
        material_params: MaterialParams {
            diffuse_colour: Vec3::ONE,
            metallic: 0.25,
            perceptual_roughness: PerceptualRoughness(0.25),
            index_of_refraction: Default::default()
        }
    };

    assert_eq!(basic_brdf(params), Vec3::splat(0.22577369));
}
