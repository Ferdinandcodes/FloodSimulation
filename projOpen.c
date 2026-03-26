/*
    Name: Ferdinand Nel
    Student Number: 0821785

    Flood forecasting simulation.
    This is the OpenMP version used to compare speed against the serial version.
*/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#define NUMROWS 8000
#define NUMCOLS 8000
#define PASSES 4
#define FLOODLEVEL 100.0
#define TIME 1.0
#define PONDING_DEPTH 5.0

typedef struct
{
    float hydraulicConductivity;
    float suctionHead;
    float moistureDeficit;
} SoilType;

/* Main arrays used by the simulation. */
float elevation[NUMROWS * NUMCOLS];
float water[NUMROWS * NUMCOLS];
int flowDirection[NUMROWS * NUMCOLS];
float totalInfiltration[NUMROWS * NUMCOLS];
float tempWater[NUMROWS * NUMCOLS];
int floodedCells[NUMROWS * NUMCOLS];
int size = NUMROWS * NUMCOLS;
float temp[NUMROWS * NUMCOLS];
int rowOffset[] = {-1, -1, -1, 0, 0, 1, 1, 1};
int colOffset[] = {-1, 0, 1, -1, 1, -1, 0, 1};

/* Fill the grid with random terrain heights. */
void generateElevations(float elevation[], int size)
{
    for (int i = 0; i < size; i++)
    {
        elevation[i] = (float)(rand() % 1000);
    }
}

/* Find all valid neighbors around one cell. */
void findNeighbors(int index, int numRows, int numCols, int *neighbors, int *numNeighbors)
{
    int row = index / numCols;
    int col = index % numCols;

    for (int i = 0; i < 8; i++)
    {
        int neighbourRow = row + rowOffset[i];
        int neighbourCol = col + colOffset[i];

        if (neighbourRow >= 0 && neighbourRow < numRows &&
            neighbourCol >= 0 && neighbourCol < numCols)
        {
            neighbors[*numNeighbors] = neighbourRow * numCols + neighbourCol;
            (*numNeighbors)++;
        }
    }
}

/* Smooth the terrain so it is less noisy. */
void smoothElevations(float elevation[], int size)
{
    for (int passes = 0; passes < PASSES; passes++)
    {
        memcpy(temp, elevation, size * sizeof(float));
        int neighbours[8];

        for (int i = 0; i < size; i++)
        {
            int numNeighbours = 0;
            findNeighbors(i, NUMROWS, NUMCOLS, neighbours, &numNeighbours);
            float sum = temp[i];
            int count = 1;

            for (int n = 0; n < numNeighbours; n++)
            {
                sum += temp[neighbours[n]];
                count++;
            }

            elevation[i] = sum / count;
        }
    }
}

/* Store the lowest neighboring cell for each cell. */
void calculateFlowDirection(float elevation[], int flowDirection[], int size)
{
    int neighbours[8];

    for (int i = 0; i < size; i++)
    {
        int numNeighbours = 0;
        findNeighbors(i, NUMROWS, NUMCOLS, neighbours, &numNeighbours);

        float lowestElevation = elevation[i];
        int lowestIndex = -1;

        for (int n = 0; n < numNeighbours; n++)
        {
            if (elevation[neighbours[n]] < lowestElevation)
            {
                lowestElevation = elevation[neighbours[n]];
                lowestIndex = neighbours[n];
            }
        }

        flowDirection[i] = lowestIndex;
    }
}

/* Add one hour of rainfall to every cell. */
void applyRainfall(float water[], float rainfall)
{
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        water[i] += rainfall;
    }
}

/* Remove some water based on soil infiltration. */
void applyInfiltration(float totalInfiltration[], float water[], SoilType soil)
{
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        float infiltrationRate = soil.hydraulicConductivity * (1 + (soil.suctionHead * soil.moistureDeficit) / totalInfiltration[i]);
        float amountInfiltrated = infiltrationRate * TIME;
        if (amountInfiltrated > water[i])
        {
            amountInfiltrated = water[i];
        }
        water[i] -= amountInfiltrated;
        totalInfiltration[i] += amountInfiltrated;
    }
}

/* Route extra water to lower neighboring cells. */
void waterRoute(float water[], int flowDirection[])
{
    (void)flowDirection;
    memset(tempWater, 0, size * sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        int neighbours[8];
        int numNeighbours = 0;
        int lowerNeighbours[8];
        int lowerCount = 0;
        float dropWeights[8];
        float totalDrop = 0.0f;
        float surface = elevation[i] + water[i];

        findNeighbors(i, NUMROWS, NUMCOLS, neighbours, &numNeighbours);

        for (int n = 0; n < numNeighbours; n++)
        {
            int neighbourIndex = neighbours[n];
            float neighbourSurface = elevation[neighbourIndex] + water[neighbourIndex];
            float drop = surface - neighbourSurface;

            if (drop > 0.0f)
            {
                lowerNeighbours[lowerCount] = neighbourIndex;
                dropWeights[lowerCount] = drop;
                totalDrop += drop;
                lowerCount++;
            }
        }

        if (lowerCount == 0 || water[i] <= 0.0f)
        {
            #pragma omp atomic
            tempWater[i] += water[i];
            continue;
        }

        float retainedWater = water[i];
        if (retainedWater > PONDING_DEPTH)
        {
            retainedWater = PONDING_DEPTH;
        }
        float movableWater = water[i] - retainedWater;

        #pragma omp atomic
        tempWater[i] += retainedWater;

        if (movableWater <= 0.0f)
        {
            continue;
        }

        for (int n = 0; n < lowerCount; n++)
        {
            int neighbourIndex = lowerNeighbours[n];
            float share = movableWater * (dropWeights[n] / totalDrop);

            #pragma omp atomic
            tempWater[neighbourIndex] += share;
        }
    }
    memcpy(water, tempWater, size * sizeof(float));
}

/* Count how many cells are flooded at the end. */
int checkFlooding(float water[], int floodedCells[])
{
    int floodedCount = 0;
    for (int i = 0; i < size; i++)
    {
        if (water[i] >= FLOODLEVEL)
        {
            floodedCells[i] = 1;
            floodedCount++;
        }
        else
        {
            floodedCells[i] = 0;
        }
    }
    printf("Simulation has ended. Total flooded cells: %d\n", floodedCount);
    printf("Flooded cells: %d out of %d (%.1f%%)\n", floodedCount, size, (float)floodedCount / size * 100);
    return floodedCount;
}

int main(void)
{
    /* Fixed seed so experiments are repeatable. */
    srand(42);
    SoilType soil;
    int soilChoice;
    int stormTime;
    float rainfall;
    FILE *simResults;
    FILE *threadData;
    FILE *speedupData;

    printf("Welcome to my HPC Project in flood forecasting\n");
    printf("The simulation is run with a %d x %d grid\n", NUMROWS, NUMCOLS);
    printf("Please choose a soil type for the simulation:\n");
    printf("1. Clay\n");
    printf("2. Loam\n");
    printf("3. Sand\n");
    printf("4. Silt\n");
    if (scanf("%d", &soilChoice) != 1)
    {
        printf("Invalid input for soil type.\n");
        return -1;
    }

    switch (soilChoice)
    {
    case 1:
        soil.hydraulicConductivity = 0.3;
        soil.suctionHead = 316.3;
        soil.moistureDeficit = 0.385;
        printf("Soil type: Clay\n");
        break;
    case 2:
        soil.hydraulicConductivity = 10.9;
        soil.suctionHead = 88.9;
        soil.moistureDeficit = 0.434;
        printf("Soil type: Loam\n");
        break;
    case 3:
        soil.hydraulicConductivity = 117.8;
        soil.suctionHead = 49.5;
        soil.moistureDeficit = 0.437;
        printf("Soil type: Sand\n");
        break;
    case 4:
        soil.hydraulicConductivity = 6.5;
        soil.suctionHead = 166.8;
        soil.moistureDeficit = 0.486;
        printf("Soil type: Silt\n");
        break;
    default:
        printf("Invalid soil type selected.\n");
        return -1;
    }

    /* Set starting values before the storm begins. */
    for (int i = 0; i < size; i++)
    {
        totalInfiltration[i] = soil.suctionHead * soil.moistureDeficit;
        water[i] = 0.0;
        flowDirection[i] = -1;
    }

    printf("Please enter the duration of the storm in hours:\n");
    if (scanf("%d", &stormTime) != 1 || stormTime <= 0)
    {
        printf("Invalid storm duration.\n");
        return -1;
    }

    printf("Please enter the rainfall intensity in mm/hr:\n");
    printf("(Light: 2-5mm, Moderate: 10-25mm, Heavy: 25-50mm, Extreme: 50-100mm+)\n");
    if (scanf("%f", &rainfall) != 1 || rainfall < 0.0f)
    {
        printf("Invalid rainfall intensity.\n");
        return -1;
    }

    /* Build terrain and downhill directions once before the storm. */
    double setupStart = omp_get_wtime();
    generateElevations(elevation, size);
    smoothElevations(elevation, size);
    calculateFlowDirection(elevation, flowDirection, size);
    double setupEnd = omp_get_wtime();
    double setupTime = setupEnd - setupStart;
    printf("Setup time: %.4f seconds\n", setupTime);

    int numThreads = omp_get_max_threads();

    /* Run the storm one hour at a time. */
    double start = omp_get_wtime();
    for (int t = 0; t < stormTime; t++)
    {
        applyRainfall(water, rainfall);
        applyInfiltration(totalInfiltration, water, soil);
        waterRoute(water, flowDirection);
    }
    double end = omp_get_wtime();
    double elapsed_time = end - start;

    /* Simple check to see the deepest water on the map. */
    float maxWater = 0.0;
    for (int i = 0; i < size; i++)
    {
        if (water[i] > maxWater)
            maxWater = water[i];
    }
    printf("Max water depth on any cell: %.2f mm\n", maxWater);

    int floodedCount = checkFlooding(water, floodedCells);
    printf("Simulation time: %.4f seconds\n", elapsed_time);
    printf("Threads used: %d\n", numThreads);

    /* Save experiment results for later graphs. */
    simResults = fopen("sim_results.dat", "a");
    if (simResults != NULL)
    {
        fprintf(simResults, "%d %d %.4f %.4f %d %.1f\n",
                NUMROWS, numThreads, setupTime, elapsed_time,
                floodedCount, (float)floodedCount / size * 100);
        fclose(simResults);
    }

    threadData = fopen("threads.dat", "a");
    if (threadData != NULL)
    {
        fprintf(threadData, "%d %d %.4f\n", NUMROWS, numThreads, elapsed_time);
        fclose(threadData);
    }

    speedupData = fopen("speedup.dat", "a");
    if (speedupData != NULL)
    {
        fprintf(speedupData, "%d %d %.4f\n", NUMROWS, numThreads, elapsed_time);
        fclose(speedupData);
    }

    return 0;
}