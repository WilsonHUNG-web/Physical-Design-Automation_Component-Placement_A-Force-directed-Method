CC			:= gcc
CXX			:= g++
LDLIBS = -lpng
MPICC			:= mpicc
MPICXX			:= mpicxx

CCFLAGS			:= -O3 -march=native -Wall -std=gnu11
CXXFLAGS		:= -O3 -march=native -Wall -std=c++0x

PFLAGS			:= -lpthread

all: ../bin/main

../bin/main: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
clean:
	rm -f ../bin/main