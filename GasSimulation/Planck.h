#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <helper_cuda.h>
#include <cutil_math.h>

/*
 * CIE Color Matching Functions (x_bar,y_bar,z_bar)
 * for wavelenghts in 5 nm increments from 380 nm to 780 nm.
 */
__device__
static const float fColorMatch[][3]={
	{0.0014f, 0.0000f, 0.0065f}, {0.0022f, 0.0001f, 0.0105f}, {0.0042f, 0.0001f, 0.0201f},
	{0.0076f, 0.0002f, 0.0362f}, {0.0143f, 0.0004f, 0.0679f}, {0.0232f, 0.0006f, 0.1102f},
	{0.0435f, 0.0012f, 0.2074f}, {0.0776f, 0.0022f, 0.3713f}, {0.1344f, 0.0040f, 0.6456f},
	{0.2148f, 0.0073f, 1.0391f}, {0.2839f, 0.0116f, 1.3856f}, {0.3285f, 0.0168f, 1.6230f},
	{0.3483f, 0.0230f, 1.7471f}, {0.3481f, 0.0298f, 1.7826f}, {0.3362f, 0.0380f, 1.7721f},
	{0.3187f, 0.0480f, 1.7441f}, {0.2908f, 0.0600f, 1.6692f}, {0.2511f, 0.0739f, 1.5281f},
	{0.1954f, 0.0910f, 1.2876f}, {0.1421f, 0.1126f, 1.0419f}, {0.0956f, 0.1390f, 0.8130f},
	{0.0580f, 0.1693f, 0.6162f}, {0.0320f, 0.2080f, 0.4652f}, {0.0147f, 0.2586f, 0.3533f},
	{0.0049f, 0.3230f, 0.2720f}, {0.0024f, 0.4073f, 0.2123f}, {0.0093f, 0.5030f, 0.1582f},
	{0.0291f, 0.6082f, 0.1117f}, {0.0633f, 0.7100f, 0.0782f}, {0.1096f, 0.7932f, 0.0573f},
	{0.1655f, 0.8620f, 0.0422f}, {0.2257f, 0.9149f, 0.0298f}, {0.2904f, 0.9540f, 0.0203f},
	{0.3597f, 0.9803f, 0.0134f}, {0.4334f, 0.9950f, 0.0087f}, {0.5121f, 1.0000f, 0.0057f},
	{0.5945f, 0.9950f, 0.0039f}, {0.6784f, 0.9786f, 0.0027f}, {0.7621f, 0.9520f, 0.0021f},
	{0.8425f, 0.9154f, 0.0018f}, {0.9163f, 0.8700f, 0.0017f}, {0.9786f, 0.8163f, 0.0014f},
	{1.0263f, 0.7570f, 0.0011f}, {1.0567f, 0.6949f, 0.0010f}, {1.0622f, 0.6310f, 0.0008f},
	{1.0456f, 0.5668f, 0.0006f}, {1.0026f, 0.5030f, 0.0003f}, {0.9384f, 0.4412f, 0.0002f},
	{0.8544f, 0.3810f, 0.0002f}, {0.7514f, 0.3210f, 0.0001f}, {0.6424f, 0.2650f, 0.0000f},
	{0.5419f, 0.2170f, 0.0000f}, {0.4479f, 0.1750f, 0.0000f}, {0.3608f, 0.1382f, 0.0000f},
	{0.2835f, 0.1070f, 0.0000f}, {0.2187f, 0.0816f, 0.0000f}, {0.1649f, 0.0610f, 0.0000f},
	{0.1212f, 0.0446f, 0.0000f}, {0.0874f, 0.0320f, 0.0000f}, {0.0636f, 0.0232f, 0.0000f},
	{0.0468f, 0.0170f, 0.0000f}, {0.0329f, 0.0119f, 0.0000f}, {0.0227f, 0.0082f, 0.0000f},
	{0.0158f, 0.0057f, 0.0000f}, {0.0114f, 0.0041f, 0.0000f}, {0.0081f, 0.0029f, 0.0000f},
	{0.0058f, 0.0021f, 0.0000f}, {0.0041f, 0.0015f, 0.0000f}, {0.0029f, 0.0010f, 0.0000f},
	{0.0020f, 0.0007f, 0.0000f}, {0.0014f, 0.0005f, 0.0000f}, {0.0010f, 0.0004f, 0.0000f},
	{0.0007f, 0.0002f, 0.0000f}, {0.0005f, 0.0002f, 0.0000f}, {0.0003f, 0.0001f, 0.0000f},
	{0.0002f, 0.0001f, 0.0000f}, {0.0002f, 0.0001f, 0.0000f}, {0.0001f, 0.0000f, 0.0000f},
	{0.0001f, 0.0000f, 0.0000f}, {0.0001f, 0.0000f, 0.0000f}, {0.0000f, 0.0000f, 0.0000f}
};

__device__
float max(float x, float y, float z) {
    float max=x;
    if(y>max) max=y;
    if(z>max) max=z;
    return max;
}
__device__
float3 blackBody(float temperature){
    float XX=0.f, YY=0.f, ZZ=0.f; /* initialize accumulators */
    float con, dis, wavelength, weight;
    short band, nbands=81;
    /* 
	 * loop over wavelength bands
     * integration by trapezoid method
     */
    for(band=0; band<nbands; band++){
        weight=((band==0)||(band==nbands-1))? 0.5f: 1.f;
		// wavelength in nm
        wavelength=380.f+(float)band*5.f; 
        // generate a black body spectrum
        con=1240.f/8.617e-5f;
		dis=3.74183e-16f*(1.f/pow(wavelength,5))/(exp(con/(wavelength*temperature))-1.f);
		// simple integration over bands
        XX+=weight*dis*fColorMatch[band][0];
        YY+=weight*dis*fColorMatch[band][1];
        ZZ+=weight*dis*fColorMatch[band][2];
    }
    // re-normalize the color scale
    float denom=max(XX, YY, ZZ);
	return {XX/denom, YY/denom, ZZ/denom};
}
__device__
uchar4 XYZ2RGB(float3 r, float3 g, float3 b, float3 color){
	float depth=255.f, gamma=0.8f, rangeMax=1.0e-10f;
	
	float den=(r.x*g.y-g.x*r.y)*b.z+(b.x*r.y-r.x*b.y)*g.z
		+(g.x*b.y-b.x*g.y)*r.z;
    float red=((color.x*g.y-g.x*color.y)*b.z+(g.x*b.y-b.x*g.y)*color.z
        +(b.x*color.y-color.x*b.y)*g.z)/den;
    float green=((r.x*color.y-color.x*r.y)*b.z+(b.x*r.y-r.x*b.y)*color.z
        +(color.x*b.y-b.x*color.y)*r.z)/den;
    float blue=((r.x*g.y-g.x*r.y)*color.z+(color.x*r.y-r.x*color.y)*g.z
        +(g.x*color.y-color.x*g.y)*r.z)/den;

    red=clamp(red, 0.f, 1.f);
	green=clamp(green, 0.f, 1.f);
	blue=clamp(blue, 0.f, 1.f);
	rangeMax=fmax(fmax(red, green), fmax(blue, rangeMax));
	red=depth*powf(red/rangeMax, gamma);
	green=depth*powf(green/rangeMax, gamma);
	blue=depth*powf(blue/rangeMax, gamma);

	return {(unsigned char)red, (unsigned char)green, (unsigned char)blue, 255u};
}
__device__
uchar4 planckColor(float temperature){
	float3 red   = {0.64f, 0.33f, 0.01f};
	float3 green = {0.29f, 0.60f, 0.11f};
	float3 blue  = {0.15f, 0.06f, 0.79f};
	return XYZ2RGB(red, green, blue, blackBody(temperature));
}