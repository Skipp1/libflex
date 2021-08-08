all:
	gcc -shared -Ofast -Wall -lm -lmvec -march=native -mtune=native -o likleyhood.so -fPIC likleyhood.c spline.c
debug:
	gcc -g -Wall -lm -lmvec -march=native -mtune=native likleyhood.c spline.c
