#! /usr/bin/env python3

# Flexknot code for fitting EDGES low band data
# Copyright (C) 2021 Henry Linton
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Lesser Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import ctypes
import matplotlib.pyplot as plt
import numpy as np
import sys


# optional requirements
try:
	import fgivenx
except:
	pass

try:
	import anesthetic
except:
	pass

# The C library goes here
ll = None

def interpolate(coef_x, coef_y, x):
	""" make sure to use the same interpolateion method as in the C

	Inputs:
	coef_x, coef_y => x and y knot locations
	x => where you want to sample

	Outputs:
	return: interpolated value at point x
	side effect: None
	"""

	d = (ctypes.c_double * len(coef_x))(*np.zeros(len(coef_x)))
	retval = (ctypes.c_double * len(x))(*np.zeros(len(x)))

	ll.spline_pchip_set(ctypes.c_size_t(len(coef_x)),
	                   (ctypes.c_double * len(coef_x))(*coef_x),
	                   (ctypes.c_double * len(coef_x))(*coef_y), d)
	ll.spline_pchip_val(ctypes.c_size_t(len(coef_x)),
	                   (ctypes.c_double * len(coef_x))(*coef_x),
	                   (ctypes.c_double * len(coef_x))(*coef_y),
	                    d,
	                    ctypes.c_size_t(len(x)),
	                   (ctypes.c_double * len(x))(*x),
	                    retval)

	return np.array(retval, dtype=np.float64)


def data_prep(samples, latex=False):
	""" prep the data by extracting it from the samples
	input: samples (from cobaya)

	output:
	name_list => the name of each fitted param
	data => np.array of the data

	side effect: None

	Hacks: accesses a private variable as cobaya doesnt tell you the names by default
	"""

	name_list = []
	data = []

	try:
		columns = samples._data.columns
	except AttributeError:
		# if we load in later as a pandas dataframe without the cobaya wrapper
		columns = samples.columns

	for param_name in columns:
		if param_name not in ['weight', 'minuslogpost', 'minuslogprior', 'minuslogprior__0', 'chi2', 'chi2__myfunction', '#']:
			if param_name[0] != 'v':
				data.append(samples[param_name])
				if latex:
					name_list.append("$"+param_name+"$")
				else:
					name_list.append(param_name)

	data = np.array(data).T

	return name_list, data


def fgx_label():
	plt.xlabel(r"Frequency $\nu$ [MHz]")
	plt.ylabel(r"Temperature $T$ [K]")
	plt.title(r"$T_{\text{data}}(\nu) - T_{\text{fg}}(\nu)$")
	return


def setrc(fullsize = True):
	""" just set a bunch of standard plotting params to make the plots pretty """
	figure = plt.gcf()
	if fullsize:
		#plt.figure(figsize=(204.12683 * 30 / 72.27 , (204.12683 * 30 / 72.27) * (9/16)))
		figure.set_size_inches(32, 18)
	else:
		figure.set_size_inches(204.12683 / 72.27 , (204.12683 / 72.27) * (3/4))
	plt.rcParams.update({
		'font.family': 'serif',
		#'font.serif': 'CMR10',
		'text.usetex': True,
		'pgf.rcfonts': False,
		'pgf.texsystem': 'pdflatex',
		'font.size': 20,
		'axes.unicode_minus': False,
		'text.latex.preamble': r'\usepackage{amsmath}'
	})

	return

def gen_coef(order, samples, bounds, fixed=None):
	""" turn samples into a set of coefs
	input:
	order => how many "free" knots (doesnt count the two fixed at each end)
	samples => the data from cobaya
	bounds => the x locations of the fixed knots at each end
	sample_no => what sample to pick from

	output: knot coefs

	side effects: None
	"""

	samples = samples.bestfit()

	coef_y = np.empty(order + 2)
	coef_x = np.empty(order + 2)

	coef_x[0] = bounds[0]
	if fixed == None:
		coef_y[0] = samples["fy_f"]
	else:
		coef_y[0] = fixed[0]

	for i in range(order):
		coef_y[i + 1] = samples["y_" + str(i + 1)]
		coef_x[i + 1] = samples["x_" + str(i + 1)]

	coef_x[-1] = bounds[1]
	if fixed == None:
		coef_y[-1] = samples["fy_l"]
	else:
		coef_y[-1] = fixed[1]

	#print(coef_x, coef_y)
	return coef_x, coef_y

def plot_fgivenx(samples, bounds, res=200, formats='contour', fixed=None):
	""" plot flexdata using fgivenx
	NOTE: Only plots the flexdata, doesnt plot any other overlayed models

	input:
	samples => from cobaya
	bounds => x locations of fixed knots
	res => resolution of fgivenx

	output: None

	side effects: pyplot is loaded
	"""

	if 'fgivenx' not in sys.modules:
		raise Exception("trying to call anesthetic, but you dont have it installed")

	name_list, data = data_prep(samples)

	print(name_list)

	# create a function for fgivenx
	coef_x_str = "[" + str(bounds[0]) +", "

	if fixed is None:
		try:
			coef_y_str = "[p[" + str(np.where(np.array(name_list) == "fy_f")[0][0]) + "], "
		except IndexError:
			print("f_yf and f_yl not foud and no \"fixed\" kwargs found. defaulting to f_yf=0, f_yl=0")
			coef_y_str = "[0, " # default to 0
	else:
		coef_y_str = "[" + str(fixed[0]) + ", "

	for i, name in enumerate(name_list):
		if name[0] == 'x':
			coef_x_str = coef_x_str + "p["+str(i)+"], "
		elif name[0] == 'y':
			coef_y_str = coef_y_str + "p["+str(i)+"], "

	coef_x_str = coef_x_str +  str(bounds[1]) + "]"

	if fixed is None:
		try:
			coef_y_str = coef_y_str + "p[" +  str(np.where(np.array(name_list) == "fy_l")[0][0]) + "] ]"
		except IndexError:
			coef_y_str = coef_y_str + "0]" # default to 0
	else:
		coef_y_str = coef_y_str + str(fixed[1]) + "]"

	#print(coef_y_str)
	#print(coef_x_str)

	interpolate_fun = "lambda x, p: interpolate("+coef_x_str+", "+coef_y_str+", x)"

	# ugly eval but idk how to get a function otherwise
	interpolate_fun = eval(interpolate_fun)


	if formats == 'line':
		fgivenx.plot_lines(interpolate_fun,
	                      np.linspace(bounds[0], bounds[1], res),
	                      data,
	                      weights=samples['weight'])
	elif formats == 'contour':
		fgivenx.plot_contours(interpolate_fun,
	                      np.linspace(bounds[0], bounds[1], res),
	                      data,
	                      weights=samples['weight'],
	                      ny=res)
	else:
		print("undefined format")
		fgivenx.plot_contours(interpolate_fun,
	                      np.linspace(bounds[0], bounds[1], res),
	                      data,
	                      weights=samples['weight'],
	                      ny=res)
	return

def plot_anesthetic(samples):
	""" plot anesthetic

	input:
	samples => from cobaya

	output: None

	side effects: pyplot is loaded
	"""
	if 'anesthetic' not in sys.modules:
		raise Exception("trying to call anesthetic, but you dont have it installed")

	name_list, data = data_prep(samples, latex=True)
	anesthetic_samples = anesthetic.samples.MCMCSamples(data=data, columns=name_list, weight=samples['weight'])
	anesthetic_samples.plot_2d(name_list)

	return


def plot_coef(order, samples, bounds, sample_no=-1):
	""" plot individual interpolation
	NOTE: Only plots the flexknot, doesnt plot any other overlayed models

	input:
	order => how many "free" knots (not counting the fixed ones)
	samples => from cobaya
	bounds => x locations of fixed knots
	sample_no => what sample to take the data from

	output: None

	side effects: pyplot is loaded
	"""

	coef_x, coef_y = gen_coef(order, samples, bounds, sample_no)
	plt.plot(coef_x, coef_y, 'rx')


if __name__ == "__main__":
	raise ImportError("libflex_plot most be imported as a module")
