CC = gcc

.PHONY : clean

heat_diffusion : heat_diffusion.o
	$(CC) -o $@ -lOpenCL heat_diffusion.o

heat_diffusion.o : heat_diffusion.c

clean :
	rm -f heat_diffusion
	rm -f heat_diffusion.o
