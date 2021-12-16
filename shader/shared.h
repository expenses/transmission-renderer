struct Uniforms {
    float3   Eye;
    int      FirstFrame;
    int2     screen_dimensions;
    float2   inverse_screen_dimensions;
    float4x4 ProjectionInverse;
    float4x4 ReprojectionMatrix;
    float4x4 ViewProjectionInverse;
    float DepthSimilaritySigma;
};

float3 DecodeOctahedron( float2 f )
{
    f = f * 2.0 - 1.0;

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
    float t = saturate( -n.z );
    n.xy += n.xy >= 0.0 ? -t : t;
    return normalize( n );
}

#define INVERTED_DEPTH_RANGE 1
