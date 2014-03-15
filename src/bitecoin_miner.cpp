#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint_client.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>

// CUDA runtime
#include <cuda_runtime.h>
// #include <curand_kernel.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define BIGINT_SIZE_BYTES 32
#define BIGINT_SIZE BIGINT_SIZE_BYTES/4
#define SALT_SIZE BIGINT_SIZE

namespace bitecoin
{

    extern bool runBitecoinMiningTrials(const size_t trialCount, const uint64_t roundId, const uint64_t roundSalt, const size_t chainDataCount, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *bestSolution, uint32_t *bestProof, size_t solutionSize, size_t trialProofSize, uint32_t *trialProofs, uint32_t *d_trialSolutions, uint32_t *d_randomSalt, uint32_t *d_trialProofs, uint8_t *d_chainData);

class cudaEndpointClient : public EndpointClient
{
    //using EndpointClient::EndpointClient; //C++11 only!

public:

	explicit cudaEndpointClient(std::string clientId,
                                std::string minerId,
                                std::unique_ptr<Connection> &conn,
                                std::shared_ptr<ILog> &log) : EndpointClient(clientId,
                                            minerId,
                                            conn,
                                            log) {};

    void MakeBid(
        const std::shared_ptr<Packet_ServerBeginRound> roundInfo,   // Information about this particular round
        const std::shared_ptr<Packet_ServerRequestBid> request,     // The specific request we received
        double period,                                                                          // How long this bidding period will last
        double skewEstimate,                                                                // An estimate of the time difference between us and the server (positive -> we are ahead)
        std::vector<uint32_t> &solution,                                                // Our vector of indices describing the solution
        uint32_t *pProof                                                                        // Will contain the "proof", which is just the value
    )
    {
        double tSafetyMargin = 0.85; // accounts for uncertainty in network conditions
        /* This is when the server has said all bids must be produced by, plus the
            adjustment for clock skew, and the safety margin
        */
        double tFinish = request->timeStampReceiveBids * 1e-9 + skewEstimate - tSafetyMargin;

        Log(Log_Verbose, "MakeBid - start, total period=%lg.", period);

        /*
            We will use this to track the best solution we have created so far.
        */
        std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
        bigint_t bestProof; //uint32_t [8];
        //set bestproof.limbs = 1's
        wide_ones(BIGINT_WORDS, bestProof.limbs);

        double worst = pow(2.0, BIGINT_LENGTH * 8); // This is the worst possible score

        //For use on GPU
        //Define some sizes for arrays
        size_t trialCount = 512;
        size_t solutionSize = sizeof(uint32_t) * roundInfo->maxIndices;
        size_t trialSolutionsSize = sizeof(uint32_t) * roundInfo->maxIndices * trialCount;
        size_t proofSize = sizeof(uint32_t) * BIGINT_SIZE;
        size_t trialProofSize = sizeof(uint32_t) * BIGINT_SIZE * trialCount;
        size_t randomSaltSize = sizeof(uint32_t) * SALT_SIZE;
        size_t chainDataSize = sizeof(uint8_t) * roundInfo->chainData.size();

        //Create CPU Buffers
        // uint32_t *trialSolutions = (uint32_t *)malloc(trialSolutionsSize);
        uint32_t *trialProofs = (uint32_t *)malloc(trialProofSize);
        uint32_t *trialSolutions = (uint32_t *)malloc(trialSolutionsSize);

        //Allocate GPU Buffers
        uint32_t *d_trialSolutions, *d_randomSalt, *d_trialProofs;
        uint8_t *d_chainData;
        // curandState *d_curand;

        checkCudaErrors(cudaMalloc((void **) &d_trialSolutions, trialSolutionsSize));
        checkCudaErrors(cudaMalloc((void **) &d_randomSalt, randomSaltSize));
        checkCudaErrors(cudaMalloc((void **) &d_chainData, chainDataSize));
        checkCudaErrors(cudaMalloc((void **) &d_trialProofs, trialProofSize));
        // checkCudaErrors(cudaMalloc((void **) &d_curand, sizeof(curandState) * trialCount));

        //Push buffers to GPU
        // checkCudaErrors(cudaMemcpy(d_trialSolutions, trialSolutions, trialSolutionsSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_randomSalt, (const void*)&roundInfo->c[0], randomSaltSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_chainData, (const void*)&roundInfo->chainData[0], chainDataSize, cudaMemcpyHostToDevice));

        unsigned nTrials = 0;
        while (1)
        {
            nTrials++;

            for (int trial = 0; trial < trialCount; trial++)
            {
                uint32_t curr = 0;
                for (int index = 0; index < roundInfo->maxIndices; index++)
                {
                    curr += 1 + (rand() % 100);
                    trialSolutions[(trial * roundInfo->maxIndices) + index] = curr;
                }
            }

            checkCudaErrors(cudaMemcpy(d_trialSolutions, trialSolutions, trialSolutionsSize, cudaMemcpyHostToDevice));

            runBitecoinMiningTrials(trialCount, roundInfo->roundId, roundInfo->roundSalt, roundInfo->chainData.size(), roundInfo->maxIndices, roundInfo->hashSteps, &bestSolution[0], &bestProof.limbs[0], solutionSize, trialProofSize, trialProofs, d_trialSolutions, d_randomSalt, d_trialProofs, d_chainData);

            double t = now() * 1e-9; // Work out where we are against the deadline
            double timeBudget = tFinish - t;
            Log(Log_Debug, "Finish trial %d, time remaining =%lg seconds.", nTrials, timeBudget);

            if (timeBudget <= 0)
                break;  // We have run out of time, send what we have
        }

        free(trialProofs);
        free(trialSolutions);

        checkCudaErrors(cudaFree(d_trialSolutions));
        checkCudaErrors(cudaFree(d_randomSalt));
        checkCudaErrors(cudaFree(d_trialProofs));
        checkCudaErrors(cudaFree(d_chainData));
        // checkCudaErrors(cudaFree(d_curand));

        solution = bestSolution;
        wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);

        Log(Log_Verbose, "MakeBid - finish.");
    }

};
};//End namespace bitecoin;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "bitecoin_client client_id logLevel connectionType [arg1 [arg2 ...]]\n");
        exit(1);
    }

    // We handle errors at the point of read/write
    signal(SIGPIPE, SIG_IGN);   // Just look at error codes

    try
    {
        std::string clientId = argv[1];
        std::string minerId = "Spunkmonkey's Miner";

        // Control how much is being output.
        // Higher numbers give you more info
        int logLevel = atoi(argv[2]);
        fprintf(stderr, "LogLevel = %s -> %d\n", argv[2], logLevel);

        std::vector<std::string> spec;
        for (int i = 3; i < argc; i++)
        {
            spec.push_back(argv[i]);
        }

        std::shared_ptr<bitecoin::ILog> logDest = std::make_shared<bitecoin::LogDest>(clientId, logLevel);
        logDest->Log(bitecoin::Log_Info, "Created log.");

        std::unique_ptr<bitecoin::Connection> connection {bitecoin::OpenConnection(spec)};

        bitecoin::cudaEndpointClient endpoint(clientId, minerId, connection, logDest);
        endpoint.Run();

    }
    catch (std::string &msg)
    {
        std::cerr << "Caught error string : " << msg << std::endl;
        return 1;
    }
    catch (std::exception &e)
    {
        std::cerr << "Caught exception : " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception." << std::endl;
        return 1;
    }

    return 0;
}

