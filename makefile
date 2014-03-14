SHELL=/bin/bash

CPPFLAGS += -std=c++11 -W -Wall  -g
CPPFLAGS += -O3
CPPFLAGS += -I include

# For your makefile, add TBB and OpenCL as appropriate

# Launch client and server connected by pipes
launch_pipes : src/bitecoin_server src/bitecoin_client
	-rm .fifo_rev
	mkfifo .fifo_rev
	# One direction via pipe, other via fifo
	src/bitecoin_client client1 3 file .fifo_rev - | (src/bitecoin_server server1 3 file - .fifo_rev &> /dev/null)

cuda_launch_pipes : src/bitecoin_server src/bitecoin_miner
	-rm .fifo_rev
	mkfifo .fifo_rev
	# One direction via pipe, other via fifo
	src/bitecoin_miner client1 3 file .fifo_rev - | (src/bitecoin_server server1 3 file - .fifo_rev &> /dev/null)

# Launch an "infinite" server, that will always relaunch
launch_infinite_server : src/bitecoin_server
	while [ 1 ]; do \
		src/bitecoin_server server1-$USER 3 tcp-server 4000; \
	done;

# Launch a client connected to a local server
connect_local : src/bitecoin_client
	src/bitecoin_client client-$USER 3 tcp-client localhost 4000

cuda_connect_local : src/bitecoin_miner
	src/bitecoin_miner client-$USER 3 tcp-client localhost 4000
	
# Launch a client connected to a shared exchange
connect_exchange : src/bitecoin_client
	src/bitecoin_client client-$(USER) 3 tcp-client $(EXCHANGE_ADDR)  $(EXCHANGE_PORT)

cuda_connect_exchange : src/bitecoin_miner
	src/bitecoin_miner funkmaster-general 3 tcp-client $(EXCHANGE_ADDR)  $(EXCHANGE_PORT)

#Build script to target evans' machine
src/test/cudaTest.o :
	nvcc -c -o src/test/cudaTest.o --compiler-options "-w" -I include/cudaInc/ src/test/cudaTest.cu

src/test/cudaTest : src/test/cudaTest.o
	g++ -g -o src/test/cudaTest -I include/cudaInc/ -L /opt/cuda/lib64/ -lcuda -lcudart src/test/cudaTest.cpp src/test/cudaTest.o
	rm src/test/cudaTest.o

rich_test_build : src/test/cudaTest
	src/test/cudaTest