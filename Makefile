#####################################################################
## Here specify the location of the IBAMR source and the location
##IBAMR_SRC_DIR = /xlwork6/heartvalve/IBAMR-2015/sfw/IBAMR
##IBAMR_BUILD_DIR = /xlwork6/heartvalve/IBAMR-2015/sfw/IBAMR/ibamr-objs-opt
##IBAMR_SRC_DIR = /xlwork6/2101013f/LiuyangIBAMR/IBAMR
##IBAMR_BUILD_DIR = /xlwork6/2101013f/LiuyangIBAMR/IBAMR/ibamr-objs-opt
##IBAMR_SRC_DIR = /xlwork6/2101013f/LiuyangIBAMR/LAE/IBAMR
##IBAMR_BUILD_DIR = /xlwork6/2101013f/LiuyangIBAMR/LAE/IBAMR/ibamr-objs-opt
#IBAMR_SRC_DIR =/work/e645/shared/sfw/ibamr/IBAMR
#IBAMR_BUILD_DIR =/work/e645/shared/sfw/ibamr/ibamr-objs-opt
IBAMR_SRC_DIR =/work/e645/shared/sfw2/ibamr_latest/IBAMR
IBAMR_BUILD_DIR =/work/e645/shared/sfw2/ibamr_latest/ibamr-objs-opt
######################################################################
## Include variables specific to the particular IBAMR build.
include $(IBAMR_BUILD_DIR)/config/make.inc

#CC=mpicc 
#CXX=mpicxx



SRC = $(wildcard *.C) $(wildcard /home/staff1/nthekkethi/FEMLDE_7/*.C)
PDIM = 3
OBJS = $(SRC:%.C=%.o) $(IBAMR_LIB_3D) $(IBTK_LIB_3D)


main: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -DNDIM=$(PDIM) -o main


# OBJS1=impedance_sub.o new_match.o f90_tools.o external_sub.o ext_new_match.o

# main: $(OBJS) $(OBJS1)
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) $(OBJS1) $(LDFLAGS) $(LIBS) -DNDIM=$(PDIM) -o main
#
#
# new_match.o: new_match.f90 f90_tools.o
# 	$(FC) -c $(FFLAGS) new_match.f90
#
# f90_tools.o: f90_tools.f90
# 	$(FC) -c $(FFLAGS) f90_tools.f90
#
# impedance_sub.o: impedance_sub.f90 f90_tools.o new_match.o
# 	$(FC) -c $(FFLAGS) impedance_sub.f90
#
# ext_new_match.o: ext_new_match.f90 f90_tools.o
# 	$(FC) -c $(FFLAGS) ext_new_match.f90
#
# external_sub.o: external_sub.f90 f90_tools.o ext_new_match.o
# 	$(FC) -c $(FFLAGS) external_sub.f90

clean:
	$(RM) main
	$(RM) *.o *.lo *.objs *.ii *.int.c *.mod
	$(RM) -r .libs

-include $(SRC:%.C=%.d)
