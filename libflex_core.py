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
import numpy as np
import cobaya

# mpi stuff
global_mpi_rank = 0
global_mpi_size = 1

# data
global_x = 0
global_y = 0

def flex_params(**kwargs):
	""" dynamically generate the config

	Input kwargs:
	order => size of the polynomial/number of knots
	knot_range => range of y locations the knot can be
	likelihood_func => function for the log likelihood
	fg_priors => foreground priors
	nlive => polychord nlive
	write => write a bunch of extra data to the disk as we go

	side effects => none
	return => cobaya config dict
	"""

	order = kwargs.get('order', 0)
	knot_range = kwargs.get('knot_range', [-1, 1])
	fg_priors = kwargs.get('fg_priors', [])
	likelihood_func = kwargs.get('likelihood_func', lambda x: print("please supply log likelihood"))
	nlive = kwargs.get('nlive', 100)
	write = kwargs.get('write', False)

	c_params = {
		"likelihood": {
			"myfunction": {
				#"external": lambda **kwargs: likelihood_func(*extract_knots(**kwargs)),
				"external": likelihood_func
			},
		},


		"params": {
			# init conditions for ordered x val
			"x_0": {
				"value": np.min(global_x),
				"drop": True,
			},
			"y_0": {
				"value": 1,
				"drop": True,
			},

			# 2 special y vals for the fixed x val at each end
			"fy_f": {
				"prior": { "min": knot_range[0], "max": knot_range[1] },
				"latex": "$fy_f$",
			},
			"fy_l": {
				"prior": { "min": knot_range[0], "max": knot_range[1] },
				"latex": "$fy_l$",
			},
		},
	}

	# speed up computation by avoiding writing to the disk as much
	if write is False:
		c_params["sampler"] = {
			"polychord": {
				"nlive": nlive,
				"write_resume": False,
				"read_resume": False,
				"write_stats": True,
				"write_live": False,
				"write_dead": False,
			},
		}
	else:
		c_params["sampler"] = {
			"polychord": {
				"nlive": nlive,
			},
		}



	# now for the params
	param_list = ["fy_f", "fy_l"]
	for i in range(order):

		# --------- for x val ordering --------------------------------------------------------------- #

		l_varx = "x_" + str(i)
		l_varv = "v_" + str(i + 1)

		# dummy variable
		c_params["params"]["v_" + str(i + 1)] = {
			"prior": {
				"dist": "uniform",
				"min": 0,
				"max": 1,
			},
			"latex": "v_" + str(i + 1),
			"drop": True,
		}

		# from arXiv/1506.00171
		param_list.append("x_" + str(i + 1))

		c_params["params"]["x_" + str(i + 1)] = {

			"value": "lambda "+l_varv+","+l_varx+": "+l_varx+"+("+str(np.max(global_x))+"-"+l_varx+")*(1-"+l_varv+"**(1/("+str(order)+"-"+str(i+1)+"+1)))",

			"min": np.min(global_x),
			"max": np.max(global_x),

			"latex": "$x_{" + str(i + 1)+"}$",
		}
		# ---------------------------------------------------------------------------------------- #

		# for regular old y val
		param_list.append("y_" + str(i + 1))

		c_params["params"]["y_" + str(i + 1)] = {
			"prior": {
				"min": knot_range[0],
				"max": knot_range[1],
				},
			"latex": "$y_{" + str(i + 1)+"}$",
		}

	# ---------------------------------------------------------------------------------------- #
	# additional foreground params
	for i in range(len(fg_priors)):
		c_params["params"]["a_" + str(i)] = {
			"prior": {
				"min": fg_priors[i][0],
				"max": fg_priors[i][1],
			},
			"latex": "$a_" + str(i)+"$",
		}
		param_list.append("a_" + str(i))

	# explicitly define the input params
	c_params["likelihood"]["myfunction"]["input_params"] = param_list

	return c_params



def run(c_params):
	""" run the sampler"""
	# run the sampler
	full_info, sampler = cobaya.run(c_params)

	# only rank 0 contains samples
	if global_mpi_rank == 0:
		samples = sampler.products()['sample']

		return full_info, sampler, samples

	else:
		# match signature on other end
		return -1, -1, -1




if __name__ == "__main__":
	raise ImportError("libflex.flex_plot most be imported as a module")


