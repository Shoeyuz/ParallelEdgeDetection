
#
#	Makefile for micha project
#

DEBUG = -g
CC = qcc
LD = qcc


TARGET = -Vgcc_ntox86_64
#TARGET = -Vgcc_ntox86
#TARGET = -Vgcc_ntoarmv7le
#TARGET = -Vgcc_ntoaarch64le


CFLAGS += $(DEBUG) $(TARGET) -Wall
LDFLAGS+= $(DEBUG) $(TARGET)
BINS = controlTower server
all: $(BINS)

clean:
	rm -f *.o $(BINS);
#	cd solutions; make clean


controlTower.o: controlTower.c controlTower.h
server.o: server.c server.h
