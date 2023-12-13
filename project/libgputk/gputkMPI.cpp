
#ifdef GPUTK_USE_MPI

#include <cstring>
#include <mpich/mpi.h>
#include <string>
#include "gputk.h"

static int rank = -1;

int gpuTKMPI_getRank() {
  if (rank != -1) {
    return rank;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int rankCount() {
  int nRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  return nRanks;
}

const char *gpuTKMPI_getStringFromRank(int rank, int tag) {
  if (isMasterQ) {
    char *buf;
    int bufSize;
    MPI_Recv(&bufSize, 1, MPI_INT, rank, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    buf = (char *)calloc(bufSize, sizeof(char));
    MPI_Recv(buf, bufSize, MPI_CHAR, rank, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    return buf;
  }

  return NULL;
}
void gpuTKMPI_sendStringToMaster(const char *str, int tag) {
  if (!isMasterQ) {
    // we are going to send the string to the master
    int len = (int)strlen(str) + 1;
    MPI_Send(&len, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    MPI_Send((void *)str, len, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }

  return;
}

int gpuTKMPI_Init(int *argc, char ***argv) {
  int err = MPI_SUCCESS;
  err     = MPI_Init(argc, argv);
  // printf("argc = %d\n", *argc);
  // err = MPI_Init(NULL, NULL);
  // printf("rank = %d is master = %d\n", gpuTKMPI_getRank(), isMasterQ);
  // MPI_Barrier(MPI_COMM_WORLD);
  return err;
}

bool finalizedQ = false;

extern "C" int gpuTKMPI_Finalize(void) {
  if (finalizedQ) {
    return MPI_SUCCESS;
  }
  finalizedQ = true;
  gpuTK_atExit();
  return MPI_Finalize();
}
extern "C" void gpuTKMPI_Exit(void) {
  gpuTKMPI_Finalize();
  return;
}

#endif /* GPUTK_USE_MPI */
