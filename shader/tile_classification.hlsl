#include "shared.h"

[[vk::binding(0, 0)]] cbuffer cbPassData {
    Uniforms uniforms;
}

[[vk::binding(1, 0)]] Texture2D<float> depth_buffer;
[[vk::binding(2, 0)]] Texture2D<float4> normal_and_velocity;

[[vk::binding(3, 0)]] Texture2D<float2> history;
[[vk::binding(4, 0)]] Texture2D<float> t2d_previousDepth;

[[vk::binding(5, 0)]] StructuredBuffer<uint>sb_raytracerResult;

[[vk::binding(6, 0)]] RWStructuredBuffer<uint> rwsb_tileMetaData;
[[vk::binding(7, 0)]] RWTexture2D<float2> reprojection_results;

[[vk::binding(8, 0)]] RWTexture2D<float3> previous_moments;
[[vk::binding(9, 0)]] RWTexture2D<float3> current_moments;

[[vk::binding(10, 0)]] SamplerState ss_trilinerClamp;

float4x4 FFX_DNSR_Shadows_GetViewProjectionInverse()
{
    return uniforms.ViewProjectionInverse;
}

float4x4 FFX_DNSR_Shadows_GetReprojectionMatrix()
{
    return uniforms.ReprojectionMatrix;
}

float4x4 FFX_DNSR_Shadows_GetProjectionInverse()
{
    return uniforms.ProjectionInverse;
}

float2 FFX_DNSR_Shadows_GetInvBufferDimensions()
{
    return uniforms.inverse_screen_dimensions;
}

int2 FFX_DNSR_Shadows_GetBufferDimensions()
{
    return uniforms.screen_dimensions;
}

int FFX_DNSR_Shadows_IsFirstFrame()
{
    return uniforms.FirstFrame;
}

float3 FFX_DNSR_Shadows_GetEye()
{
    return uniforms.Eye;
}

float FFX_DNSR_Shadows_ReadDepth(int2 p)
{
    return depth_buffer.Load(int3(p, 0)).x;
}

float FFX_DNSR_Shadows_ReadPreviousDepth(int2 p)
{
    return t2d_previousDepth.Load(int3(p, 0)).x;
}

float3 FFX_DNSR_Shadows_ReadNormals(int2 p)
{
    return DecodeOctahedron(normal_and_velocity.Load(int3(p, 0)).xy);
}

float2 FFX_DNSR_Shadows_ReadVelocity(int2 p)
{
    float4 n_and_v = normal_and_velocity.Load(int3(p, 0));
    return n_and_v.zw;
}

float FFX_DNSR_Shadows_ReadHistory(float2 p)
{
    return history.SampleLevel(ss_trilinerClamp, p, 0).x;
}

float3 FFX_DNSR_Shadows_ReadPreviousMomentsBuffer(int2 p)
{
    return previous_moments.Load(p).xyz;
}

uint  FFX_DNSR_Shadows_ReadRaytracedShadowMask(uint p)
{
    return sb_raytracerResult[p];
}

void  FFX_DNSR_Shadows_WriteMetadata(uint p, uint val)
{
    rwsb_tileMetaData[p] = val;
}

void  FFX_DNSR_Shadows_WriteMoments(uint2 p, float3 val)
{
    current_moments[p] = val;
}

void FFX_DNSR_Shadows_WriteReprojectionResults(uint2 p, float2 val)
{
    reprojection_results[p] = val;
}

bool FFX_DNSR_Shadows_IsShadowReciever(uint2 p)
{
    float depth = FFX_DNSR_Shadows_ReadDepth(p);
    return (depth > 0.0f) && (depth < 1.0f);
}

#include "../external/FidelityFX-Denoiser/ffx-shadows-dnsr/ffx_denoiser_shadows_tileclassification.h"

[numthreads(8, 8, 1)]
void tile_classification(uint group_index : SV_GroupIndex, uint2 gid : SV_GroupID)
{
    /*if (gid.x == 0 && gid.y == 0 && group_index == 0) {
        printf(
            "======================\nEye: vec3(%v3f),\nFirstFrame: %u\n screen dimensionss: %v2u\n inv screen dimensions: %v2f\n piv: %v4f\nrr: %v4f\n",
            uniforms.Eye, uniforms.FirstFrame, uniforms.screen_dimensions, uniforms.inverse_screen_dimensions, uniforms.ProjectionInverse[0],
            uniforms.ReprojectionMatrix[0]
        );
    }*/
    FFX_DNSR_Shadows_TileClassification(group_index, gid);
}
