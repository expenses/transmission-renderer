#include "shared.h"

[[vk::binding(0, 0)]] cbuffer cbPassData {
    Uniforms uniforms;
}

[[vk::binding(1, 0)]] Texture2D<float> depth_buffer;
[[vk::binding(2, 0)]] Texture2D<float4> normal_and_velocity;

// Consists of float2(mean, variance);
[[vk::binding(3, 0)]] Texture2D<float2> history;
[[vk::binding(4, 0)]] Texture2D<float> t2d_previousDepth;

[[vk::binding(5, 0)]] StructuredBuffer<uint> sb_raytracerResult;

[[vk::binding(6, 0)]] RWStructuredBuffer<uint> rwsb_tileMetaData;
[[vk::binding(7, 0)]] RWTexture2D<float2> reprojection_results;

[[vk::binding(8, 0)]] RWTexture2D<float3> previous_moments;
[[vk::binding(9, 0)]] RWTexture2D<float3> current_moments;

[[vk::binding(10, 0)]] SamplerState ss_trilinerClamp;

[[vk::binding(11, 0)]] RWTexture2D<float2> rw_history;

[[vk::binding(0, 1)]] RWTexture2D<unorm float4> rwt2d_output;


float2 FFX_DNSR_Shadows_GetInvBufferDimensions()
{
    return uniforms.inverse_screen_dimensions;
}

int2 FFX_DNSR_Shadows_GetBufferDimensions()
{
    return uniforms.screen_dimensions;
}

float4x4 FFX_DNSR_Shadows_GetProjectionInverse()
{
    return uniforms.ProjectionInverse;
}

float FFX_DNSR_Shadows_GetDepthSimilaritySigma()
{
    return uniforms.DepthSimilaritySigma;
}

float FFX_DNSR_Shadows_ReadDepth(int2 p)
{
    return depth_buffer.Load(int3(p, 0));
}

float16_t3 FFX_DNSR_Shadows_ReadNormals(int2 p)
{
    return (float16_t3) DecodeOctahedron(normal_and_velocity.Load(int3(p, 0)).xy);
}

bool FFX_DNSR_Shadows_IsShadowReciever(uint2 p)
{
    float depth = FFX_DNSR_Shadows_ReadDepth(p);
    return (depth > 0.0f) && (depth < 1.0f);
}

float16_t2 FFX_DNSR_Shadows_ReadInput(int2 p)
{
    return (float16_t2)reprojection_results.Load(p).xy;
}

uint FFX_DNSR_Shadows_ReadTileMetaData(uint p)
{
    return rwsb_tileMetaData[p];
}

#include "../external/FidelityFX-Denoiser/ffx-shadows-dnsr/ffx_denoiser_shadows_filter.h"

[numthreads(8, 8, 1)]
void filter_pass_0(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID)
{
    const uint PASS_INDEX = 0;
    const uint STEP_SIZE = 1;

    bool bWriteOutput = false;
    float2 const results = FFX_DNSR_Shadows_FilterSoftShadowsPass(gid, gtid, did, bWriteOutput, PASS_INDEX, STEP_SIZE);

    if (bWriteOutput)
    {
        rw_history[did] = results;
    }
}

[numthreads(8, 8, 1)]
void filter_pass_1(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID)
{
    const uint PASS_INDEX = 1;
    const uint STEP_SIZE = 2;

    bool bWriteOutput = false;
    float2 const results = FFX_DNSR_Shadows_FilterSoftShadowsPass(gid, gtid, did, bWriteOutput, PASS_INDEX, STEP_SIZE);
    if (bWriteOutput)
    {
        rw_history[did] = results;
    }
}


float ShadowContrastRemapping(float x)
{
    const float a = 10.f;
    const float b = -1.0f;
    const float c = 1 / pow(2, a);
    const float d = exp(-b);
    const float e = 1 / (1 / pow((1 + d), a) - c);
    const float m = 1 / pow((1 + pow(d, x)), a) - c;

    return m * e;
}

[numthreads(8, 8, 1)]
void filter_pass_2(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID)
{
    const uint PASS_INDEX = 2;
    const uint STEP_SIZE = 4;

    bool bWriteOutput = false;
    float2 const results = FFX_DNSR_Shadows_FilterSoftShadowsPass(gid, gtid, did, bWriteOutput, PASS_INDEX, STEP_SIZE);

    // Recover some of the contrast lost during denoising
    const float shadow_remap = max(1.2f - results.y, 1.0f);
    const float mean = pow(abs(results.x), shadow_remap);

    if (bWriteOutput)
    {
        rwt2d_output[did].x = mean;
    }
}
