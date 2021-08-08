/*
 * Flexknot code for fitting EDGES low band data
 * Copyright (C) 2021 Henry Linton
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Lesser Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "spline.h" //  https://people.sc.fsu.edu/~jburkardt/c_src/spline/spline.html
#include "likleyhood.h"

#define PI 3.141592653589793
#define global_border 0.1 // padding to prevent fgivenx from getting confused


struct global_data {
	double *x;
	double *y;
	size_t len;
	size_t order;
} global_data;

struct pchip_buffer {
	double *d;
	double *out;
} pchip_buffer;

struct coef {
	double *x;
	double *y;
	double *a;
};

void init_globals(double *x, double *y, size_t len, size_t order) {
	 /* I must be called to init the data buffers at the start
	 * and cleanup must be called at the end to clear out the
	 * allocated memory
	 *
	 * Inputs:
	 * x => array of len, containing the data x values
	 * y => array of len, containing the data y values
	 * len => length of x and y arrays
	 * order => number of knots (not counting the two fixed at each end of the data
	 *          I.E. if order=0, then you have a linear fit)
	 *
	 * Return: void
	 *
	 * Side Effects:
	 * global_data.x gets allocated and initialised
	 * global_data.y gets allocated and initialised
	 * global_data.len gets initialised
	 * global_data.order gets initialised
	 *
	 * pchip_buffer.d gets allocated
	 * pchip_buffer.out gets allocated
	 *
	 */
	global_data.x = (double *)malloc(sizeof(double) * len);
	global_data.y = (double *)malloc(sizeof(double) * len);

	for(int i=0; i < len; i++) {
		global_data.x[i] = x[i];
		global_data.y[i] = y[i];
	}
	global_data.len = len;
	global_data.order = order;

	pchip_buffer.d = (double *)malloc(sizeof(double) * (global_data.order+2));
	pchip_buffer.out = (double *)malloc(sizeof(double) * (global_data.len));
	return;
}

/////////////// Start T21 foreground //////////////////////////////////////////////

double t21fg_edgesa(double *a, double nu) {
	/*
	 * foregound model used in the EDGES paper
	 */
	double nuc = 75.0;
	return a[0] * pow((nu / nuc), -2.5)                             // use -a[0] to match the edges plot
	     + a[1] * pow((nu / nuc), -2.5) * log(nu / nuc)
	     + a[2] * pow((nu / nuc), -2.5) * pow(log(nu / nuc), 2.0)
	     + a[3] * pow((nu / nuc), -4.5)
			 + a[4] * pow((nu / nuc), -2.0);                            //  use -a[4] to match the edges plot
}

double t21fg_sims(double *d, double nu) {
	/*
	 * foregound model used by  sims and pober
	 */

	double nuc = 75.0;
	double Tcal = pow((nu/nuc), d[0]) * (d[2]*sin(2 * PI * nu / d[1]) + d[3]*cos(2 * PI * nu / d[1]));
	double Tfg_pow = 0;
	for (int i=4; i < 4+5; i++) {
		Tfg_pow += pow(10, d[i] * pow(log10(nu/nuc), i));
	}
	return  Tfg_pow + Tcal;
}

double t21fg(double *a, double nu) {
	/*
	 * wrapper for the above models to facilitate
	 * easy switching without changing code in multiple places
	 */
	return t21fg_edgesa(a, nu);
}

/////////////// End T21 foreground //////////////////////////////////////////////

struct coef kwargs2coef(char** keys, double *val, size_t len) {
	/* allocate the coefs depending on what they are named
	 * E.G. the arg named y_1 gets allocated to coef_y, etc
	 *
	 * At the moment there are 4 different cases, however more can be added
	 * case 'x' => the x location of the knots
	 * case 'y' => the y location of the knots
	 * case 'a' => additional coefs for use in T21 foregound model (or anything else)
	 * case 'f' => the y location of the two knots at each end of the data
	 *
	 * Inputs:
	 * keys => an array of strings, each one containing the name of the coef
	 * val => an array of values for that coef
	 * len => length of the array
	 *
	 * Return:
	 * Struct coef containing all the coefs allocated
	 *
	 * Side effects:
	 * struct coef has memory allocated *** MEMORY LEAK DANGER ***
	 *
	 */

	struct coef coef;
	coef.x = (double *)malloc(sizeof(double) * (global_data.order+2));
	coef.y = (double *)malloc(sizeof(double) * (global_data.order+2));
	coef.a = (double *)malloc(sizeof(double) * (len-(2*global_data.order+2)));

	int x_loc = 1;
	int y_loc = 1;
	int a_loc = 0;
	int f_loc = 0;

	for (int i=0; i < len; i++) {
		switch(keys[i][0]) {

			case 'x':
				coef.x[x_loc] = val[i];
				x_loc++;
				break;

			case 'y':
				coef.y[y_loc] = val[i];
				y_loc++;
				break;

			case 'a':
				coef.a[a_loc] = val[i];
				a_loc++;
				break;

			case 'f':
			// the two fixed x/y vals at each end
				if(f_loc == 0) {
					coef.x[0] = global_data.x[0]-global_border;
					coef.y[0] = val[i];
					f_loc++;
				}
				else {
					coef.x[global_data.order+1] = global_data.x[global_data.len-1]+global_border;
					coef.y[global_data.order+1] = val[i];
				}
				break;
			}
		}
	return coef;
}

double logpdf(double y, double mean) {
	/* log pdf
	 *
	 * Input:
	 * y => where we want to look at
	 * mean => mean of dist
	 *
	 * Return: logpdf
	 * Side Effects: None
	 *
	 */
	const double stdev = 0.025; // Gaussian error from the edges data

	double u = (y - mean) / stdev;
	return (-0.5 * pow(u, 2)) - log(sqrt(2 * PI) * stdev);
}


double log_likleyhood(char** keys, double *val, size_t len) {
	/* This is the function that is wrapped in python then passed to cobaya
	 *
	 * Inputs:
	 * keys => an array of strings, each one containing the name of the coef
	 * val => an array of values for that coef
	 * len => length of the array
	 *
	 * Return: the logpdf of all coef and x values
	 *
	 * Side effects: None
	 *
	 */
	struct coef coef = kwargs2coef(keys, val, len);

	double retval = 0;

	spline_pchip_set(global_data.order+2, coef.x, coef.y, pchip_buffer.d);
	spline_pchip_val(global_data.order+2, coef.x, coef.y, pchip_buffer.d, global_data.len, global_data.x, pchip_buffer.out);

	for (int i=0; i < global_data.len; i++){
		retval += logpdf(global_data.y[i], pchip_buffer.out[i] + t21fg(coef.a, global_data.x[i]));
	}
	free(coef.x);
	free(coef.y);
	free(coef.a);
	return retval;
}

void cleanup(void) {
	/* cleanup function to deallocate all the memory used */
	free(global_data.x);
	free(global_data.y);
	free(pchip_buffer.d);
	free(pchip_buffer.out);
	return;
}
