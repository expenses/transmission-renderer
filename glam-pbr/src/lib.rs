#![no_std]

use core::f32::consts::{FRAC_1_PI, PI};
use glam::{Mat4, Vec2, Vec3};
use num_traits::Float;

// Workarounds: can't use f32.lerp, f32.clamp or f32.powi.

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

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

    let alpha_roughness_sq = roughness.0 * roughness.0;

    let f = (noh * noh) * (alpha_roughness_sq - 1.0) + 1.0;

    alpha_roughness_sq / (PI * f * f)
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

    let ggx = ggx_v + ggx_l;

    if ggx > 0.0 {
        0.5 / ggx
    } else {
        0.0
    }
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

    fn apply_ior(self, ior: IndexOfRefraction) -> PerceptualRoughness {
        PerceptualRoughness(self.0 * clamp(ior.0 * 2.0 - 2.0, 0.0, 1.0))
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

pub fn transmission_btdf(
    material_params: MaterialParams,
    normal: Normal,
    view: View,
    light: Light,
) -> Vec3 {
    let actual_roughness = material_params.perceptual_roughness.as_actual_roughness();
    let index_of_refraction = material_params.index_of_refraction;

    let transmission_roughness = actual_roughness.apply_ior(index_of_refraction);

    let light_mirrored = Light((light.0 + 2.0 * normal.0 * (-light.0).dot(normal.0)).normalize());

    let halfway = Halfway::new(&view, &light_mirrored);
    let normal_dot_halfway = Dot::new(&normal, &halfway);
    let view_dot_halfway = Dot::new(&view, &halfway);
    let normal_dot_view = Dot::new(&normal, &view);
    let normal_dot_light_mirrored = Dot::new(&normal, &light_mirrored);

    let distribution = d_ggx(normal_dot_halfway, transmission_roughness);

    let geometric_shadowing = v_smith_ggx_correlated(
        normal_dot_view,
        normal_dot_light_mirrored,
        transmission_roughness,
    );

    let f0 = {
        Vec3::splat(index_of_refraction.to_dielectric_f0())
            .lerp(material_params.diffuse_colour, material_params.metallic)
    };
    let f90 = Vec3::ONE;

    let fresnel = fresnel_schlick(view_dot_halfway, f0, f90);

    (1.0 - fresnel) * distribution * geometric_shadowing * material_params.diffuse_colour
}

pub struct IblVolumeRefractionParams {
    pub material_params: MaterialParams,
    pub framebuffer_size_x: u32,
    pub normal: Normal,
    pub view: View,
    pub proj_view_matrix: Mat4,
    pub position: Vec3,
    pub thickness: f32,
    pub model_scale: f32,
}

fn refract(incident: Vec3, normal: Vec3, index_of_refraction: IndexOfRefraction) -> Vec3 {
    let eta = 1.0 / index_of_refraction.0;

    let n_dot_i = normal.dot(incident);

    let k = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);

    eta * incident - (eta * n_dot_i + k.sqrt()) * normal
}

fn get_volume_transmission_ray(
    normal: Normal,
    view: View,
    thickness: f32,
    index_of_refraction: IndexOfRefraction,
    scale: f32,
) -> Vec3 {
    let refraction = refract(-view.0, normal.0, index_of_refraction);
    // todo: work out a better instancing scheme so we don't have to hardcode this.
    let gltf_primitive_scale = 0.25;
    refraction.normalize() * thickness * scale * gltf_primitive_scale
}

pub fn ibl_volume_refraction<
    FSamp: Fn(Vec2, f32) -> Vec3,
    GSamp: Fn(f32, PerceptualRoughness) -> Vec2,
>(
    params: IblVolumeRefractionParams,
    framebuffer_sampler: FSamp,
    ggx_lut_sampler: GSamp,
) -> Vec3 {
    let IblVolumeRefractionParams {
        framebuffer_size_x,
        proj_view_matrix,
        position,
        normal,
        view,
        thickness,
        model_scale,
        material_params:
            MaterialParams {
                diffuse_colour: base_colour,
                metallic: _,
                perceptual_roughness,
                index_of_refraction,
            },
    } = params;

    //let thickness = 1.0;
    //let perceptual_roughness = PerceptualRoughness(0.25);

    let refracted_ray_exit = position
        + get_volume_transmission_ray(normal, view, thickness, index_of_refraction, model_scale);

    let device_coords = proj_view_matrix * refracted_ray_exit.extend(1.0);
    let screen_coords = Vec2::new(device_coords.x, device_coords.y) / device_coords.w;
    let texture_coords = (screen_coords + 1.0) / 2.0;

    let framebuffer_lod =
        (framebuffer_size_x as f32).log2() * perceptual_roughness.apply_ior(index_of_refraction).0;

    let transmitted_light = framebuffer_sampler(texture_coords, framebuffer_lod);
    // todo: volume
    let attenuated_colour = transmitted_light;

    let normal_dot_view = normal.0.dot(view.0);
    let brdf = ggx_lut_sampler(normal_dot_view, perceptual_roughness);

    let f0 = index_of_refraction.to_dielectric_f0();
    let f90 = Vec3::ONE;

    let specular_colour = f0 * brdf.x + f90 * brdf.y;

    (1.0 - specular_colour) * attenuated_colour * base_colour
}

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

pub fn basic_brdf(params: BasicBrdfParams) -> BrdfResult {
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
    let f0 = { Vec3::splat(index_of_refraction.to_dielectric_f0()).lerp(diffuse_colour, metallic) };

    let fresnel = fresnel_schlick(view_dot_halfway, f0, Vec3::splat(1.0));

    let diffuse = light_intensity * normal_dot_light.value * diffuse_brdf(c_diff, fresnel);
    let specular = light_intensity
        * normal_dot_light.value
        * specular_brdf(
            normal_dot_view,
            normal_dot_light,
            normal_dot_halfway,
            actual_roughness,
            fresnel,
        );

    BrdfResult {
        diffuse,
        specular,
        emission: Vec3::ZERO,
    }
}

#[derive(Default)]
pub struct BrdfResult {
    pub diffuse: Vec3,
    pub specular: Vec3,
    pub emission: Vec3,
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
            index_of_refraction: Default::default(),
        },
    };

    assert_eq!(basic_brdf(params), Vec3::splat(0.22577369));
}
