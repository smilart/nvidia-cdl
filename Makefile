PROGRAM=nvidia-cdl

$(PROGRAM):
	
	/usr/local/cuda/bin/nvcc -c -o $(PROGRAM).o $(PROGRAM).cu -I./
	/usr/local/cuda/bin/nvcc $(PROGRAM).o -lnvidia-ml -o $(PROGRAM) -lcudart_static

clean:
	rm -f $(PROGRAM) $(PROGRAM).o
