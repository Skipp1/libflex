
#ifndef LIKLEYHOOD_H
#define LIKLEYHOOD_H

void init_globals(double *x, double *y, size_t len, size_t order);
double log_likleyhood(char** keys, double *val, size_t len);
double t21fg(double *a, double nu);
void cleanup(void);

#endif //LIKLEYHOOD_H
