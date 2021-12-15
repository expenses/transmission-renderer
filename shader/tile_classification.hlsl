struct FFX_DNSR_Shadows_Data_Defn {
    float3   Eye;
    int      FirstFrame;
    int2     BufferDimensions;
    float2   InvBufferDimensions;
    float4x4 ProjectionInverse;
    float4x4 ReprojectionMatrix;
    float4x4 ViewProjectionInverse;
};

[[vk::binding(0, 0)]] cbuffer cbPassData {
    FFX_DNSR_Shadows_Data_Defn FFX_DNSR_Shadows_Data;
}

[[vk::binding(1, 0)]] Texture2D<float> depth_buffer;
[[vk::binding(2, 0)]] Texture2D<float4> normal_and_velocity;

[[vk::binding(3, 0)]] Texture2D<float> t2d_history;
[[vk::binding(4, 0)]] Texture2D<float> t2d_previousDepth;

[[vk::binding(5, 0)]] StructuredBuffer<uint>sb_raytracerResult;

[[vk::binding(6, 0)]] RWStructuredBuffer<uint> rwsb_tileMetaData;
[[vk::binding(7, 0)]] RWTexture2D<float2>         rwt2d_reprojectionResults;

[[vk::binding(8, 0)]] Texture2D<float3>           t2d_previousMoments;
[[vk::binding(9, 0)]] RWTexture2D<float3>         rwt2d_momentsBuffer;

[[vk::binding(10, 0)]] SamplerState ss_trilinerClamp;

float4x4 FFX_DNSR_Shadows_GetViewProjectionInverse()
{
    return FFX_DNSR_Shadows_Data.ViewProjectionInverse;
}

float4x4 FFX_DNSR_Shadows_GetReprojectionMatrix()
{
    return FFX_DNSR_Shadows_Data.ReprojectionMatrix;
}

float4x4 FFX_DNSR_Shadows_GetProjectionInverse()
{
    return FFX_DNSR_Shadows_Data.ProjectionInverse;
}

float2 FFX_DNSR_Shadows_GetInvBufferDimensions()
{
    return FFX_DNSR_Shadows_Data.InvBufferDimensions;
}

int2 FFX_DNSR_Shadows_GetBufferDimensions()
{
    return FFX_DNSR_Shadows_Data.BufferDimensions;
}

int FFX_DNSR_Shadows_IsFirstFrame()
{
    return FFX_DNSR_Shadows_Data.FirstFrame;
}

float3 FFX_DNSR_Shadows_GetEye()
{
    return FFX_DNSR_Shadows_Data.Eye;
}

float FFX_DNSR_Shadows_ReadDepth(int2 p)
{
    return depth_buffer.Load(int3(p, 0)).x;
}

float FFX_DNSR_Shadows_ReadPreviousDepth(int2 p)
{
    return t2d_previousDepth.Load(int3(p, 0)).x;
}

float3 DecodeOctahedron( float2 f )
{
    f = f * 2.0 - 1.0;

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
    float t = saturate( -n.z );
    n.xy += n.xy >= 0.0 ? -t : t;
    return normalize( n );
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
    return t2d_history.SampleLevel(ss_trilinerClamp, p, 0);
}

float3 FFX_DNSR_Shadows_ReadPreviousMomentsBuffer(int2 p)
{
    return t2d_previousMoments.Load(int3(p, 0)).xyz;
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
    rwt2d_momentsBuffer[p] = val;
}

void FFX_DNSR_Shadows_WriteReprojectionResults(uint2 p, float2 val)
{
    rwt2d_reprojectionResults[p] = val;
}

bool FFX_DNSR_Shadows_IsShadowReciever(uint2 p)
{
    float depth = FFX_DNSR_Shadows_ReadDepth(p);
    return (depth > 0.0f) && (depth < 1.0f);
}

#define INVERTED_DEPTH_RANGE 1
#include "../external/FidelityFX-Denoiser/ffx-shadows-dnsr/ffx_denoiser_shadows_tileclassification.h"

[numthreads(8, 8, 1)]
void tile_classification(uint group_index : SV_GroupIndex, uint2 gid : SV_GroupID)
{
    FFX_DNSR_Shadows_TileClassification(group_index, gid);
}
