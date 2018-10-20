CC = gcc

.PHONY clean

heat_diffusion :
	$(CC) -o $@ heat_diffusion.c

clean :
	rm -f heat_diffusion
