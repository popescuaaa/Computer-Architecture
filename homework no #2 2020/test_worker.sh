#!/bin/bash

#------------ Homework number 2 --------------- #
#       Andrei Gabriel Popescu 333CA            #
#                                               #
#-----------------------------------------------#

# Load modules                                  
module load compilers/gnu-5.4.0 

# Generate the executables
make

#------------- Execution for BLAS ---------------------
#
chmod 777 ./tema2_blas
chmod 777 ./tema2_neopt
chmod 777 ./tema2_opt_m

# Test blas flavour
./tema2_blas input &> blas.out

#------------------- Crectness -------------------------
./compare out1 /export/asc/tema2/out1 0.01
./compare out2 /export/asc/tema2/out2 0.01
./compare out3 /export/asc/tema2/out3 0.01

# ------------- Execution for NEOPT --------------------
#
# Test neopt flavour
./tema2_neopt input &> neopt.out

#------------------- Crectness -------------------------
./compare out1 /export/asc/tema2/out1 0.01
./compare out2 /export/asc/tema2/out2 0.01
./compare out3 /export/asc/tema2/out3 0.01

# ------------- Execution for OPT_M --------------------
#
# Test opt_m flavour
./tema2_opt_m input &> opt_m.out

#------------------- Crectness -------------------------
./compare out1 /export/asc/tema2/out1 0.01
./compare out2 /export/asc/tema2/out2 0.01
./compare out3 /export/asc/tema2/out3 0.01

# ------------- Execution for OPT_F --------------------
#
# Test opt_m flavour
./tema2_opt_f input &> opt_f.out

#------------------- Crectness -------------------------
./compare out1 /export/asc/tema2/out1 0.01
./compare out2 /export/asc/tema2/out2 0.01
./compare out3 /export/asc/tema2/out3 0.01

# ------------- Execution for OPT_F_EXTRA --------------
#
# Test opt_m flavour
./tema2_opt_f_extra input &> opt_f_extra.out

#------------------- Crectness -------------------------
./compare out1 /export/asc/tema2/out1 0.01
./compare out2 /export/asc/tema2/out2 0.01
./compare out3 /export/asc/tema2/out3 0.01
