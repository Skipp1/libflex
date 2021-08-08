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
import libflex_core as libflex
import libflex_plot as flexplot
import matplotlib.pyplot as plt
import sys
import ctypes
from mpi4py import MPI

# mpi setup
global_mpi_comm = MPI.COMM_WORLD
global_mpi_rank = global_mpi_comm.Get_rank()
global_mpi_size = global_mpi_comm.Get_size()

# ctypes definitions
ll = ctypes.CDLL("./likleyhood.so")

ll.log_likleyhood.restype  =    ctypes.c_double
ll.log_likleyhood.argtypes =   [ctypes.POINTER(ctypes.c_char_p),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_size_t]

ll.init_globals.argtypes =     [ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_size_t,
                                ctypes.c_size_t]

ll.spline_pchip_set.argtypes = [ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double)]

ll.spline_pchip_val.argtypes = [ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double)]

def likelihood_wrap(**kwargs):
	# this is just
	# ll.log_likleyhood(keys, values, len)
	# but with extra steps to convert to ctypes
	return ll.log_likleyhood((ctypes.c_char_p * (len(kwargs)+1))(*np.array([*kwargs.keys()]).astype(bytes), None),
	                         (ctypes.c_double * (len(kwargs)))(*kwargs.values()),
	                          ctypes.c_size_t(len(kwargs)))


def main():

	"""entrypoint"""

	data = np.genfromtxt("figure1_plotdata.csv", skip_header=1, delimiter=",")[3:-2].T
	x_val = data[0]
	y_val = data[2]

	libflex.global_x = x_val
	libflex.global_y = y_val

	libflex.global_mpi_rank = global_mpi_rank
	libflex.global_mpi_size = global_mpi_size

	order = int(sys.argv[1])

	ll.init_globals((ctypes.c_double * (len(x_val)))(*x_val),
	                (ctypes.c_double * (len(y_val)))(*y_val),
	                 ctypes.c_size_t(len(x_val)),
	                 ctypes.c_size_t(order))


	# foreground priors
	#fg_priors = [[-40000, -20000], [-25000, -10000], [-7000, -3000], [250, 500], [20000, 40000]] # Low
	#fg_priors = [[-10000, 10000], [-10000, 5000], [-3000, 1000], [100, 250], [-10000, 15000]] # High
	fg_priors = [[-100000, 100000]]*5 # unconstrained

	# gen config & run
	c_params = libflex.flex_params(order=order,
	                               knot_range = [-1, 1],
	                               fg_priors = fg_priors,
	                               likelihood_func = likelihood_wrap,
	                               nlive=100,
	                               write=False)

	# lock each end
	#c_params["params"]["fy_f"] = {"value": 0}
	#c_params["params"]["fy_l"] = {"value": 0}


	# save output
	#c_params["output"] = "chains/"+str(order)

	full_info, sampler, samples = libflex.run(c_params)

	# single threaded bit
	if global_mpi_rank == 0:

		# print the bestfit
		bf = samples.bestfit()
		print(bf)

		# give plot access to the C lib
		flexplot.ll = ll

		#---------------------- FGIVENX ------------------------#
		flexplot.setrc(fullsize=True)
		flexplot.fgx_label()
		try:
			flexplot.plot_fgivenx(samples, [np.min(x_val), np.max(x_val)], res=300, formats='line', fixed = [c_params["params"]["fy_f"]["value"], c_params["params"]["fy_l"]["value"]])
		except:
			flexplot.plot_fgivenx(samples, [np.min(x_val), np.max(x_val)], res=300, formats='line')
		plt.savefig("out/"+str(order)+"-fgx-lines.pdf", bbox_inches='tight')
		plt.clf()


		flexplot.setrc(fullsize=True)
		flexplot.fgx_label()
		try:
			flexplot.plot_fgivenx(samples, [np.min(x_val), np.max(x_val)], res=300, formats='noline', fixed = [c_params["params"]["fy_f"]["value"], c_params["params"]["fy_l"]["value"]])
		except:
			flexplot.plot_fgivenx(samples, [np.min(x_val), np.max(x_val)], res=300, formats='noline')
		plt.savefig("out/"+str(order)+"-fgx-contour.pdf", bbox_inches='tight')
		plt.clf()


		#---------------------- anesthetic ------------------------#
		flexplot.plot_anesthetic(samples)
		flexplot.setrc(fullsize=True)
		plt.savefig("out/"+str(order)+"-ane.pdf", bbox_inches='tight')
		plt.clf()


	ll.cleanup()
	return


if __name__ == "__main__":
	main()
