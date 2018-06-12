ARGET := kernels

CC = g++

OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)


CPP_FLAGS=-std=gnu++11

ifeq ($(debug), 1)
	CPP_FLAGS=-g -O0 -fno-inline
endif


LIBS =

INCLUDE = -I.\
		  -I/usr/include/eigen3

.PHONY: all clean fpga

fpga: CL_INCLUDE = $(shell aocl compile-config)
fpga: CL_LIBS    = $(shell aocl link-config | sed 's/lterasic/laltera/g')
fpga: $(TARGET)

all: fpga

%.o: %.cpp $(HEADERS)
	$(CC) -c $< -o $@ $(CPP_FLAGS) $(INCLUDE) $(CL_INCLUDE)

$(TARGET): $(OBJECTS)
	$(CC) $(CPP_FLAGS) $(OBJECTS) -Wall $(LIBS) $(CL_LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
