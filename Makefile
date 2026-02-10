CFLAGS := -Wall -Wextra -Werror -Wfatal-errors -O3 -ffast-math -mtune=native -march=native

.PHONY: all
all: libNICE.a

libNICE.a: NICE_Kernels.o NICE.o
	$(AR) rcs $@ $^

NICE.o: NICE.c NICE.h NICE_Kernels.h

NICE_Kernels.o: NICE_Kernels.c NICE_Kernels.h
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY:
clean:
	$(RM) *.o *.a
