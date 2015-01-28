VPATH = src include
CC = nvcc
INC_FLAGS = -I./
INC_FLAGS += -Iinclude
INC_FLAGS += -I/usr/apps.sp3/cuda/6.0/samples/common/inc 
DEBUG = -arch=sm_35
CFLAGS = $(DEBUG)
#LIB_PATH = 
LIBS = -lm
TARGET_CVA = cva
TARGET_CVA_DELTA = cva_delta

SOURCES_CVA  = src/calc_cva.cu
SOURCES_CVA += src/default_probability.cu
SOURCES_CVA += src/discount_factors.cu
SOURCES_CVA += src/main_cva.cu
SOURCES_CVA += src/mt19937.cu
SOURCES_CVA += src/random_generator.cu
SOURCES_CVA += src/short_rates.cu
SOURCES_CVA += src/spot_FXs.cu
SOURCES_CVA += src/swaps.cu
SOURCES_CVA += src/swaptions.cu
SOURCES_CVA += src/utilities.cu
SOURCES_CVA += src/zero_coupon_bonds.cu

OBJS_CVA  = calc_cva.o
OBJS_CVA += default_probability.o
OBJS_CVA += discount_factors.o
OBJS_CVA += main_cva.o
OBJS_CVA += mt19937.o
OBJS_CVA += random_generator.o
OBJS_CVA += short_rates.o
OBJS_CVA += spot_FXs.o
OBJS_CVA += swaps.o
OBJS_CVA += swaptions.o
OBJS_CVA += utilities.o
OBJS_CVA += zero_coupon_bonds.o

SOURCES_CVA_DELTA  = src/calc_cva_delta.cu
SOURCES_CVA_DELTA += src/default_probability.cu
SOURCES_CVA_DELTA += src/discount_factors.cu
SOURCES_CVA_DELTA += src/main_cva_delta.cu
SOURCES_CVA_DELTA += src/mt19937.cu
SOURCES_CVA_DELTA += src/random_generator.cu
SOURCES_CVA_DELTA += src/short_rates.cu
SOURCES_CVA_DELTA += src/spot_FXs.cu
SOURCES_CVA_DELTA += src/swaps.cu
SOURCES_CVA_DELTA += src/swaptions.cu
SOURCES_CVA_DELTA += src/utilities.cu
SOURCES_CVA_DELTA += src/zero_coupon_bonds.cu

OBJS_CVA_DELTA  = calc_cva_delta.o
OBJS_CVA_DELTA += default_probability.o
OBJS_CVA_DELTA += discount_factors.o
OBJS_CVA_DELTA += main_cva_delta.o
OBJS_CVA_DELTA += mt19937.o
OBJS_CVA_DELTA += random_generator.o
OBJS_CVA_DELTA += short_rates.o
OBJS_CVA_DELTA += spot_FXs.o
OBJS_CVA_DELTA += swaps.o
OBJS_CVA_DELTA += swaptions.o
OBJS_CVA_DELTA += utilities.o
OBJS_CVA_DELTA += zero_coupon_bonds.o

SELF = basic.mk

.SUFFIXES :
.SUFFIXES : .o .c
.c.o :
	$(CC) $(CFLAGS) $(INC_FLAGS) -c $<

.SUFFIXES : .o .cu
.cu.o :
	$(CC) $(CFLAGS) $(INC_FLAGS) -c $<

all: $(TARGET_CVA_DELTA)
#all: $(TARGET_CVA) $(TARGET_CVA_DELTA)

$(TARGET_CVA): $(OBJS_CVA)
	$(CC) $(CFLAGS) $(INC_FLAGS) $(LIB_PATH) $(LIBS) -o $@ $(OBJS_CVA)

$(TARGET_CVA_DELTA): $(OBJS_CVA_DELTA)
	$(CC) $(CFLAGS) $(INC_FLAGS) $(LIB_PATH) $(LIBS) -o $@ $(OBJS_CVA_DELTA)

Makefile :${FRC} $(SELF)
	rm -f $@
	cp $(SELF) $@
	chmod +w $@
	echo '# Automatically-generated dependencies list:' >> $@
	${CC} ${CFLAGS} $(INC_FLAGS) -M $(SOURCES_CVA) >> $@
	chmod -w $@

.PHONY: clean
clean :
	rm -f *.o $(TARGET_EM)  $(TARGET_CVA) $(TARGET_CVA_DELTA)


