/*
 *      RGB VALUES FOR HOT OBJECTS   by Dan Bruton (astro@sfasu.edu)
 *
 *      This program can be found at
 *           http://www.physics.sfasu.edu/astro/color.html
 *      and was last updated on March 19, 1996.
 *
 *      Reference Book
 *      Color Science : Concepts and Methods, Quantitative Data and Formula
 *                      by Gunter Wyszecki, W. S. Stiles
 *                      John Wiley & Sons; ISBN: 0471021067
 *
 *      This program will calculate the RGB values for a given
 *      energy distribution as a function of wavelength of light.
 *      A black body is used and an example.
 *
 *      NetPBM's ppmtogif can be used to convert the ppm image generated
 *      to a gif.  The red, green and blue values (RGB) are
 *      assumed to vary linearly with wavelength (for GAM=1).
 *      NetPBM Software: ftp://ftp.cs.ubc.ca/ftp/archive/netpbm/
 ***
 *      Converted to C by William T. Bridgman (bridgman@wyeth.gsfc.nasa.gov)
 *      in February, 2000.
 *        - Original integration method replaced by trapezoid rule.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Function prototypes */
void BlackBody(double temperature, double *X, double *Y,double *Z);
void TickMark(double temperature, short iType,double *R, double *G, double *B);
void XYZ2RGB(double xr, double yr, double zr, double xg, double yg, double zg,
              double xb, double yb, double zb, double xc, double yc, double zc,
              double *red, double *green,double *blue);
double DMAX1(double x, double y, double z);

int main(int argc,char** argv) {

    static short    colorValues[600][100][3]; /* image array */

/*
 *      Chromaticity Coordinates for Red, Green, Blue and White
 */
    double  XRed=0.64, YRed=0.33;
    double  XGreen=0.29, YGreen=0.60;
    double  XBlue=0.15, YBlue=0.06;
    double  XWhite=0.3127, YWhite=0.3291;
    double  ZRed, ZGreen, ZBlue, ZWhite;
    double  X, Y, Z, red, green, blue;
	/*
	 * IMAGE INFO - WIDTH, HEIGHT, DEPTH, GAMMA
	 */
    long imageWidth=600, imageHeight=50, colorDepth=255;
    short iType;
    long i, j;

    double gamma=0.8, colorMax, temperature, rangeMax;

    FILE *PPMFile;

    ZRed=1.0-(XRed+YRed);
    ZGreen=1.0-(XGreen+YGreen);
    ZBlue=1.0-(XBlue+YBlue);
    ZWhite=1.0-(XWhite+YWhite);
	/*
	 * FIND COLOR VALUE, colorValues, OF EACH PIXEL
	 */
    colorMax=0.0;

    /* loop over temperature range */
    for(i=0;i<imageWidth;i++) {
        temperature=1000.0 + (float)i * (10000.0/imageWidth);
        iType=2;
        BlackBody(temperature,&X,&Y,&Z); /* X,Y,Z */
        XYZ2RGB(XRed,YRed,ZRed,XGreen,YGreen,ZGreen,XBlue,YBlue,ZBlue,
                      X,Y,Z,&red,&green,&blue); /* convert to RGB */

        /* draw a 'line' of the color bar */
        for(j=0;j<imageHeight;j++) {
            /* draw tick mark if necessary */
            if (j>40) TickMark(temperature,iType,&red,&green,&blue);

            rangeMax=1.0e-10;
			rangeMax=fmax(fmax(red, green), fmax(blue, rangeMax));
			
			colorValues[i][j][0]=(short)(float)(colorDepth)*pow(red/rangeMax,gamma);
			colorValues[i][j][1]=(short)(float)(colorDepth)*pow(green/rangeMax,gamma);
			colorValues[i][j][2]=(short)(float)(colorDepth)*pow(blue/rangeMax,gamma);
        }

        if(DMAX1(red,green,blue)>colorMax) colorMax=DMAX1(red,green,blue);

    }

/*
 *****************************************************************************
 *
 *      WRITE OUTPUT TO PPM FILE
 *
 *      Output color values are normalized to color depth.
 */
    if((fopen_s(&PPMFile,"temp.ppm","w+"))!=0) {
       printf("Cannot open file.\n");
       exit(1);
    }
    fprintf(PPMFile,"P3\n");
    fprintf(PPMFile,"# temp.ppm\n");
    fprintf(PPMFile,"%d %d\n",imageWidth,imageHeight);
    fprintf(PPMFile,"%d\n",colorDepth);
    for(j=0;j<imageHeight;j++){
		for(i=0;i<imageWidth;i++){
			fprintf(PPMFile,"%d %d %d\t",colorValues[i][j][0],
        colorValues[i][j][1],colorValues[i][j][2]);
		}
		fprintf(PPMFile,"\n");
	}
    fclose(PPMFile);
    return 0;
}
/*
 ****************************************************************************
 *
 *      XYZ VALUES FROM TEMPERATURE OF OBJECT
 *
 *       A black body approximation is used where the temperature,
 *       T, is given in Kelvin.  The XYZ values are determined by
 *      "integrating" the product of the wavelength distribution of
 *       energy and the XYZ functions for a uniform source.
 */
void BlackBody(double temperature, double *X, double *Y,double *Z) {

/*
 *      CIE Color Matching Functions (x_bar,y_bar,z_bar)
 *      for wavelenghts in 5 nm increments from 380 nm to 780 nm.
 */
       double fColorMatch[][3]={
         {0.0014, 0.0000, 0.0065},
         {0.0022, 0.0001, 0.0105},
         {0.0042, 0.0001, 0.0201},
         {0.0076, 0.0002, 0.0362},
         {0.0143, 0.0004, 0.0679},
         {0.0232, 0.0006, 0.1102},
         {0.0435, 0.0012, 0.2074},
         {0.0776, 0.0022, 0.3713},
         {0.1344, 0.0040, 0.6456},
         {0.2148, 0.0073, 1.0391},
         {0.2839, 0.0116, 1.3856},
         {0.3285, 0.0168, 1.6230},
         {0.3483, 0.0230, 1.7471},
         {0.3481, 0.0298, 1.7826},
         {0.3362, 0.0380, 1.7721},
         {0.3187, 0.0480, 1.7441},
         {0.2908, 0.0600, 1.6692},
         {0.2511, 0.0739, 1.5281},
         {0.1954, 0.0910, 1.2876},
         {0.1421, 0.1126, 1.0419},
         {0.0956, 0.1390, 0.8130},
         {0.0580, 0.1693, 0.6162},
         {0.0320, 0.2080, 0.4652},
         {0.0147, 0.2586, 0.3533},
         {0.0049, 0.3230, 0.2720},
         {0.0024, 0.4073, 0.2123},
         {0.0093, 0.5030, 0.1582},
         {0.0291, 0.6082, 0.1117},
         {0.0633, 0.7100, 0.0782},
         {0.1096, 0.7932, 0.0573},
         {0.1655, 0.8620, 0.0422},
         {0.2257, 0.9149, 0.0298},
         {0.2904, 0.9540, 0.0203},
         {0.3597, 0.9803, 0.0134},
         {0.4334, 0.9950, 0.0087},
         {0.5121, 1.0000, 0.0057},
         {0.5945, 0.9950, 0.0039},
         {0.6784, 0.9786, 0.0027},
         {0.7621, 0.9520, 0.0021},
         {0.8425, 0.9154, 0.0018},
         {0.9163, 0.8700, 0.0017},
         {0.9786, 0.8163, 0.0014},
         {1.0263, 0.7570, 0.0011},
         {1.0567, 0.6949, 0.0010},
         {1.0622, 0.6310, 0.0008},
         {1.0456, 0.5668, 0.0006},
         {1.0026, 0.5030, 0.0003},
         {0.9384, 0.4412, 0.0002},
         {0.8544, 0.3810, 0.0002},
         {0.7514, 0.3210, 0.0001},
         {0.6424, 0.2650, 0.0000},
         {0.5419, 0.2170, 0.0000},
         {0.4479, 0.1750, 0.0000},
         {0.3608, 0.1382, 0.0000},
         {0.2835, 0.1070, 0.0000},
         {0.2187, 0.0816, 0.0000},
         {0.1649, 0.0610, 0.0000},
         {0.1212, 0.0446, 0.0000},
         {0.0874, 0.0320, 0.0000},
         {0.0636, 0.0232, 0.0000},
         {0.0468, 0.0170, 0.0000},
         {0.0329, 0.0119, 0.0000},
         {0.0227, 0.0082, 0.0000},
         {0.0158, 0.0057, 0.0000},
         {0.0114, 0.0041, 0.0000},
         {0.0081, 0.0029, 0.0000},
         {0.0058, 0.0021, 0.0000},
         {0.0041, 0.0015, 0.0000},
         {0.0029, 0.0010, 0.0000},
         {0.0020, 0.0007, 0.0000},
         {0.0014, 0.0005, 0.0000},
         {0.0010, 0.0004, 0.0000},
         {0.0007, 0.0002, 0.0000},
         {0.0005, 0.0002, 0.0000},
         {0.0003, 0.0001, 0.0000},
         {0.0002, 0.0001, 0.0000},
         {0.0002, 0.0001, 0.0000},
         {0.0001, 0.0000, 0.0000},
         {0.0001, 0.0000, 0.0000},
         {0.0001, 0.0000, 0.0000},
         {0.0000, 0.0000, 0.0000}};

    double XX=0.0, YY=0.0, ZZ=0.0; /* initialize accumulators */
    double con, dis, wavelength, weight;
    short band, nbands=81;

    /* loop over wavelength bands
     * integration by trapezoid method
     */
    for(band=0; band<nbands; band++) {
        weight=1.0;
        if((band==0)||(band==nbands-1)) weight=0.5; /* properly weight end points */
        wavelength=380.0+(double)band*5.0;/* wavelength in nm */
        /* generate a black body spectrum */
        con=1240.0/8.617e-5;

dis=3.74183e-16*(1.0/pow(wavelength,5))/(exp(con/(wavelength*temperature))-1.);
/* simple integration over bands */
        XX=XX+weight*dis*fColorMatch[band][0];
        YY=YY+weight*dis*fColorMatch[band][1];
        ZZ=ZZ+weight*dis*fColorMatch[band][2];
    } /* end of 'band' loop */

    /* re-normalize the color scale */
    *X=XX/DMAX1(XX,YY,ZZ);
    *Y=YY/DMAX1(XX,YY,ZZ);
    *Z=ZZ/DMAX1(XX,YY,ZZ);

}

/*****************************************************************************
 *
 *      PLACE MARKERS ON IMAGE
 */
void TickMark(double temperature, short iType,
    double *red, double *green, double *blue) {

    long k;
/*
 *      ITYPE=1 - PLAIN IMAGE
 *      ITYPE=2 - MARK IMAGE AT 1000 K INTEVALS
 */
    if(iType==2) {
        for(k=1000;k<=10000;k+=1000) {
            if(abs((long)temperature-k)<=1) {
                *red=0.0;
                *green=0.0;
                *blue=0.0;
            }
        }
    }
}

/*
 *********************************************************************
 */
void XYZ2RGB(double xr, double yr, double zr,
              double xg, double yg, double zg,
              double xb, double yb, double zb,
              double xColor, double yColor, double zColor,
              double *red, double *green,double *blue) {

    double denominator;

    denominator=(xr*yg-xg*yr)*zb+(xb*yr-xr*yb)*zg+(xg*yb-xb*yg)*zr;

    *red=((xColor*yg-xg*yColor)*zb+(xg*yb-xb*yg)*zColor
        +(xb*yColor-xColor*yb)*zg)/denominator;
    *green=((xr*yColor-xColor*yr)*zb+(xb*yr-xr*yb)*zColor
        +(xColor*yb-xb*yColor)*zr)/denominator;
    *blue=((xr*yg-xg*yr)*zColor+(xColor*yr-xr*yColor)*zg
        +(xg*yColor-xColor*yg)*zr)/denominator;

    if(*red<0.0) *red=0.0;
    if(*red>1.0) *red=1.0;
    if(*green<0.0) *green=0.0;
    if(*green>1.0) *green=1.0;
    if(*blue<0.0) *blue=0.0;
    if(*blue>1.0) *blue=1.0;
}

/* ************************************************************************/
double DMAX1(double x, double y, double z) {
    double max;
    max=x;
    if(y>max) max=y;
    if(z>max) max=z;
    return max;
}
