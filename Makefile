CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall -Iinc
TYPES = sequential mpi mpi_sync

.PHONY: all lib clean

all: lib

lib: $(addsuffix .a, $(addprefix lib/knnring_, $(TYPES)))

lib/%.a: %.o
	ar rcs $@ $<

knnring_sequential.o: src/knnring_sequential.c
	$(CC) $(CFLAGS) -o $@ -c $<

knnring_mpi.o: src/knnring_mpi.c
	$(MPICC) $(CFLAGS) -o $@ -c $<

knnring_mpi_sync.o: src/knnring_mpi_sync.c
	$(MPICC) $(CFLAGS) -o $@ -c $<

clean:
	rm *.o lib/knnring*.a
