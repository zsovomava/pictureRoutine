
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include "Header.cuh"
#include <cMath>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

__global__ void negateKernel(unsigned char* dev_in, unsigned char* dev_out, int width, int height, int widthStep, int channels)
{
    // A szálak (threads) koordinátáinak kiszámítása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szálak koordinátái a képként megadott tartományon belül vannak-e
    if (x < width && y < height) {
        // A pixel indexének kiszámítása a bemeneti képen
        int pixel_index = (width * y * channels) + y * widthStep + x * channels;

        // Végigiterálunk a színcsatornákon (RGB esetén 3 csatorna)
        for (int c = 0; c < channels; c++) {
            // A pixelek színértékeinek negálása (255 - pixel_value)
            dev_out[pixel_index + c] = 255 - dev_in[pixel_index + c]; // Negálás
        }
    }
}

__global__ void changeLookUpKernel(unsigned char* dev_in, unsigned char* dev_out, int width, int height, int widthStep, int channels, unsigned char* dev_lookUp) {
    // A szálak koordinátáinak meghatározása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szálak a képtartományon belül helyezkednek-e el
    if (x < width && y < height) {
        // A pixel indexének kiszámítása a bemeneti képen
        int pixel_index = (width * y * channels) + y * widthStep + x * channels;

        // Végigiterálunk a színcsatornákon (pl. RGB esetén 3 csatorna)
        for (int c = 0; c < channels; c++) {
            // Kép átalakítása a look-up tábla alapján
            // A dev_in érték alapján megkeresi a dev_lookUp táblában az új értéket, és dev_out-ba menti
            dev_out[pixel_index + c] = dev_lookUp[dev_in[pixel_index + c]];
        }
    }
}

__global__ void grayKernel(unsigned char* dev_in, unsigned char* dev_out, int width, int height, int widthStep, int channels) {
    // A szálak koordinátáinak meghatározása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szálak a képtartományon belül vannak-e
    if (x < width && y < height) {
        // A bemeneti pixel indexének kiszámítása, figyelembe véve a csatornákat
        int pixel_index = (width * y * channels) + y * widthStep + x * channels;

        // A kimeneti kép egycsatornás, ezért itt új pixel index számítódik kevesebb csatornával
        int newPixel_index = width * y + y * widthStep + x;

        // Szürkeárnyalatos konverzió a bemeneti képből, az RGB súlyozott összeadásával
        // A szürke értéket a Rec. 601 szabvány alapján számítjuk: 0.299 * R + 0.587 * G + 0.114 * B
        int tmp = (dev_in[pixel_index] * 299 + dev_in[pixel_index + 1] * 587 + dev_in[pixel_index + 2] * 114) / 1000;

        // Az eredmény tárolása a kimeneti szürkeárnyalatos képben
        dev_out[newPixel_index] = tmp;
    }
}

__global__ void histogramKernel(unsigned char* dev_grayIn, int* dev_Out, int width, int height, int widthStep, int channels) {
    // A szálak koordinátáinak meghatározása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szálak a képtartományon belül helyezkednek-e el
    if (x < width && y < height) {
        // A pixel indexének kiszámítása a szürkeárnyalatos képen
        int pixel_index = width * y + y * widthStep + x;

        // A hisztogramhoz hozzáadunk egyet a megfelelő árnyalatú pixel számához
        // A `dev_grayIn[pixel_index]` az adott pixel szürkeárnyalatát adja meg (0-255 közötti érték)
        // Az `atomicAdd` biztosítja, hogy egyszerre csak egy szál módosíthatja a dev_Out tömböt, elkerülve az adatütközéseket
        atomicAdd(&dev_Out[dev_grayIn[pixel_index]], 1);
    }
}

__global__ void MaskKernel(unsigned char* dev_in, unsigned char* dev_out, int width, int height, int widthStep, int channels, int* matrix, int matrix_x, int matrix_y) {
    // A szálak koordinátáinak meghatározása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szál a képtartományon belül van-e
    if (x < width && y < height) {
        long divisor = 0; // Az összegzési mátrix összegző értéke (normalizáláshoz)
        long sum = 0;     // A szűrt pixelérték összegző változója

        // Számoljuk ki a mátrix középpontját
        int half_matrix_x = matrix_x / 2;
        int half_matrix_y = matrix_y / 2;

        // Végigiterálunk a mátrix elemein
        for (int i = -half_matrix_x; i <= half_matrix_x; i++) {
            int x_in_picture = x + i; // Számoljuk a mátrix egyes elemeinek megfelelő képpont X koordinátáját

            for (int j = -half_matrix_y; j <= half_matrix_y; j++) {
                int y_in_picture = y + j; // Számoljuk a mátrix elemeinek megfelelő képpont Y koordinátáját

                // Biztosítjuk, hogy a koordináták a képen belül legyenek
                if (x_in_picture >= 0 && x_in_picture < width && y_in_picture >= 0 && y_in_picture < height) {
                    // A mátrix aktuális értékének lekérése
                    int matrix_value = matrix[(i + half_matrix_x) * matrix_y + (j + half_matrix_y)];
                    divisor += matrix_value; // A normáláshoz szükséges osztó értékhez hozzáadjuk a mátrix elem értékét

                    // Bemeneti pixel index kiszámítása
                    int pixel_index = (width * y_in_picture) + y_in_picture * widthStep + x_in_picture;
                    // A mátrix érték és a pixel érték szorzatának hozzáadása az összeghez
                    sum += dev_in[pixel_index] * matrix_value;
                }
            }
        }

        // A kimeneti pixel érték kiszámítása
        int output_pixel_index = width * y + y * widthStep + x;
        divisor = divisor != 0 ? divisor : 1; // Ha a divisor nulla, beállítjuk 1-re (elkerülve a nullával való osztást)
        dev_out[output_pixel_index] = abs(sum / divisor); // Az eredményt a kimeneti képre írjuk
    }
}

__global__ void SobelKernel(unsigned char* dev_gray_x, unsigned char* dev_gray_y, unsigned char* dev_out, int width, int height, int widthStep) {
    // A szálak koordinátáinak meghatározása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szál a képtartományon belül helyezkedik-e el
    if (x < width && y < height) {
        // Az aktuális pixel indexének kiszámítása a bemeneti képen
        int idx = y * width + y * widthStep + x;

        // A gradiens nagyságának kiszámítása a Sobel-operátor X és Y kimeneti értékeiből
        // A dev_gray_x és dev_gray_y a kép éleinek Sobel-szűrésével nyert gradiens komponensek X és Y irányban
        float gradient_magnitude = sqrtf(dev_gray_x[idx] * dev_gray_x[idx] + dev_gray_y[idx] * dev_gray_y[idx]);

        // Érintési küszöbérték alkalmazása: ha a gradiens nagysága nagyobb, mint 100, fekete (0),
        // ha kisebb, fehér (255) lesz az érték a kimeneti képen (dev_out)
        dev_out[idx] = (gradient_magnitude > 100) ? 0 : 255;
    }
}

__global__ void LaplaceKernel(unsigned char* dev_out, int width, int height, int widthStep) {
    // A szálak koordinátáinak meghatározása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szál a képtartományon belül helyezkedik-e el
    if (x < width && y < height) {
        // Az aktuális pixel indexének kiszámítása
        int idx = y * width + y * widthStep + x;

        // Laplace eredményének négyzetét vesszük, és egy küszöbértéket alkalmazunk
        // Ha a (dev_out[idx])^2 nagyobb, mint 50, akkor fehér (255), különben fekete (0)
        dev_out[idx] = (dev_out[idx] * dev_out[idx]) > 50 ? 255 : 0;
    }
}

__global__ void SumAreaKernel(unsigned char* dev_in, unsigned int* dev_sum, int width, int height, int widthStep, int dims) {
    // A szálak koordinátáinak meghatározása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szál a képtartományon belül helyezkedik-e el
    if (x < width && y < height) {
        // Az aktuális pixel indexének kiszámítása
        int idx = y * width + y * widthStep + x; // Bemeneti pixel index
        int idx_sum = y * width + x; // Összegkép pixel index
        int half_dim = dims / 2; // A terület felének mérete

        // Végigiterálunk a szomszédos pixeleken a megadott területen belül
        for (int i = -half_dim; i <= half_dim; i++) {
            int x_in_picture = x + i; // Szomszédos pixel X koordinátája

            for (int j = -half_dim; j <= half_dim; j++) {
                int y_in_picture = y + j; // Szomszédos pixel Y koordinátája

                // Ellenőrizzük, hogy a szomszédos pixel a képen belül van-e
                if (x_in_picture >= 0 && x_in_picture < width && y_in_picture >= 0 && y_in_picture < height) {
                    // Hozzáadjuk a szomszédos pixel értékét az összegképhez
                    dev_sum[y * width + x] += dev_in[x_in_picture + y_in_picture * (width + widthStep)];
                }
            }
        }
    }
}

__global__ void KLTKernel(unsigned char* dev_gray_in, unsigned char* dev_out, unsigned int* dev_derival_xy, unsigned int* dev_Doublederival_x, unsigned int* dev_Doublederival_y, int width, int height, int widthStep, double k, int th) {
    // A szálak koordinátáinak meghatározása
    int x = blockIdx.x * blockDim.x + threadIdx.x; // X koordináta
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Y koordináta

    // Ellenőrizzük, hogy a szál a képtartományon belül helyezkedik-e el
    if (x >= width || y >= height) return;

    // Az aktuális pixel indexének kiszámítása
    int idx = y * width + x;

    // Kiszámítjuk a derivált mátrix A, B, és C komponenseit
    unsigned int A = dev_Doublederival_x[idx];
    unsigned int B = dev_Doublederival_y[idx];
    unsigned int C = dev_derival_xy[idx];

    // Harris-KLT sarokdetektálási metrika kiszámítása
    double R = ((A * B - C * C) + k * (A + B) * (A + B)) / 100000;

    // Színes képkimenet pixel indexe
    int channels = 3;
    int gidx = y * widthStep + y * width + x; // Szürkeárnyalatos index
    int cidx = (width * y * channels) + y * widthStep + x * channels; // Színes kimeneti index

    // Az eredeti szürkeárnyalatos pixelérték beállítása a színes képen minden csatornára
    dev_out[cidx] = dev_gray_in[gidx];
    dev_out[cidx + 1] = dev_gray_in[gidx];
    dev_out[cidx + 2] = dev_gray_in[gidx];

    // Sarokdetektálási küszöb alkalmazása
    if (R > th) {
        // Ha a küszöbértéket meghaladja, jelöljük ki a sarkot kék színnel a kimeneti képen
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                // Ellenőrizzük, hogy a környező pixel a képen belül van-e
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nidx = ny * width * 3 + ny * widthStep + nx * 3;
                    dev_out[nidx] = 0;       // Kék szín R komponense
                    dev_out[nidx + 1] = 0;   // Kék szín G komponense
                    dev_out[nidx + 2] = 255; // Kék szín B komponense
                }
            }
        }
    }
}


extern "C" __declspec(dllexport) void RunNegateKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels)
{
    unsigned char* dev_in;
    unsigned char* dev_out;
    // Számítsuk ki a szükséges memória méretét
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);

    // Eszközmemória lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, size);

    // A bemeneti kép másolása az eszköz memóriájába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);

    // Blokk és rács méretének beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Kernel futtatása
    negateKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    // Kimeneti kép átmásolása a gazdagép memóriájába
    cudaMemcpy(pictureOut, dev_out, size, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_in);
    cudaFree(dev_out);
}
extern "C" __declspec(dllexport) void RungammaKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, float gamma) {
    unsigned char* dev_in;
    unsigned char* dev_out;
    unsigned char* dev_lookUp;
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);

    // Lookup tábla létrehozása a gamma korrekcióhoz
    unsigned char lookUp[256];
    for (size_t i = 0; i < 256; i++) {
        float normalizedValue = static_cast<float>(i) / 255.0f; // Normalizálás
        unsigned char correctedValue = static_cast<unsigned char>(255 * pow(normalizedValue, gamma));
        lookUp[i] = correctedValue; // Gamma korrekció
    }

    // Eszközmemória lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, size);
    cudaMalloc((void**)&dev_lookUp, 256 * sizeof(unsigned char));

    // Másolások az eszközmemóriába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lookUp, lookUp, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Blokk és rács méretének beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Kernel futtatása a gamma korrekcióhoz
    changeLookUpKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels, dev_lookUp);

    cudaDeviceSynchronize();

    // Kimeneti kép átmásolása a gazdagép memóriájába
    cudaMemcpy(pictureOut, dev_out, size, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_lookUp);
}
extern "C" __declspec(dllexport) void RunLogKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, float c) {
    unsigned char* dev_in;
    unsigned char* dev_out;
    unsigned char* dev_lookUp;
    // Számítsuk ki a szükséges memória méretét
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);

    // Lookup tábla létrehozása a logaritmikus korrekcióhoz
    unsigned char lookUp[256];
    for (size_t i = 0; i < 256; i++) {
        float log = c * std::log(1 + i); // Logaritmikus érték számítása
        lookUp[i] = static_cast<char>(std::min(255.0f, log)); // Érték korlátozása 255-re
    }

    // Eszközmemória lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, size);
    cudaMalloc((void**)&dev_lookUp, 256 * sizeof(char));

    // Másolások az eszközmemóriába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lookUp, lookUp, 256 * sizeof(char), cudaMemcpyHostToDevice);

    // Blokk és rács méretének beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Kernel futtatása a logaritmikus korrekcióhoz
    changeLookUpKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels, dev_lookUp);

    cudaDeviceSynchronize();

    // Kimeneti kép átmásolása a gazdagép memóriájába
    cudaMemcpy(pictureOut, dev_out, size, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_lookUp);
}
extern "C" __declspec(dllexport) void RunGrayKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels) {
    unsigned char* dev_in;
    unsigned char* dev_out;
    // Számítsuk ki a szükséges memória méretét
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    // Eszközmemória lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, graySize);

    // A bemeneti kép másolása az eszköz memóriájába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);

    // Blokk és rács méretének beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Kernel futtatása a szürkeárnyalatú konvertáláshoz
    grayKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels);

    cudaDeviceSynchronize();

    // Kimeneti kép átmásolása a gazdagép memóriájába
    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_in);
    cudaFree(dev_out);
}
extern "C" __declspec(dllexport) void RunHistogramKernel(unsigned char* pictureIn, int* histogramOut, int width, int height, int widthStep, int channels) {
    unsigned char* dev_out;
    unsigned char* dev_in;
    int* dev_histogramOut;

    // Számítsuk ki a szükséges memória méretét
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    // Eszközmemória lefoglalása a bemeneti képhez, kimeneti képhez és a hisztogramhoz
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_histogramOut, 256 * sizeof(int));

    // Másoljuk a bemeneti képet az eszköz memóriájába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_histogramOut, histogramOut, 256 * sizeof(int), cudaMemcpyHostToDevice);

    // Blokk és rács méretének beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Szürkeárnyalatú konvertálás
    grayKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    // Hisztogram számítása
    histogramKernel << <grid, block >> > (dev_out, dev_histogramOut, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    // Kimeneti hisztogram átmásolása a gazdagép memóriájába
    cudaMemcpy(histogramOut, dev_histogramOut, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_out);
    cudaFree(dev_histogramOut);
    cudaFree(dev_in);
}
extern "C" __declspec(dllexport) void RunHistogramEqualizationKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels) {
    unsigned char* dev_in;
    unsigned char* dev_out;
    int* dev_histogramOut;
    int* histogramOut = (int*)malloc((width * height + widthStep * height) * sizeof(unsigned char)); // Hisztogram tárolására
    unsigned char lookUp[256]; // Lookup tábla a hisztogram egyenlősítéshez
    unsigned char* dev_lookUp;

    // Számítsuk ki a szükséges memória méretét
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    // Inicializáljuk a hisztogramot nullára
    for (size_t i = 0; i < 256; i++)
    {
        histogramOut[i] = 0;
    }

    // Eszközmemória lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_histogramOut, 256 * sizeof(int));
    cudaMalloc((void**)&dev_lookUp, 256 * sizeof(unsigned char));

    // Másoljuk a bemeneti képet az eszköz memóriájába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_histogramOut, histogramOut, 256 * sizeof(int), cudaMemcpyHostToDevice);

    // Blokk és rács méretének beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Szürkítés
    grayKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    // Hisztogram számítása
    histogramKernel << <grid, block >> > (dev_out, dev_histogramOut, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    // Kimeneti hisztogram átmásolása a gazdagép memóriájába
    cudaMemcpy(histogramOut, dev_histogramOut, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Lookup tábla létrehozása az egyenlősítéshez
    int cumulateSum = 0; // Akkumulált összeg
    long pixelNumbers = width * height; // Összes pixel száma
    for (size_t i = 0; i < 256; i++)
    {
        cumulateSum += histogramOut[i]; // Akumultáljuk a hisztogram értékeit
        lookUp[i] = 255 * ((double)cumulateSum) / pixelNumbers; // Lookup érték számítása
    }

    // Másoljuk a lookup táblát az eszköz memóriájába
    cudaMemcpy(dev_lookUp, lookUp, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Kép egyenlősítése a lookup tábla segítségével
    changeLookUpKernel << <grid, block >> > (dev_out, dev_out, width, height, widthStep, 1, dev_lookUp);

    cudaDeviceSynchronize();

    // Kimeneti kép átmásolása a gazdagép memóriájába
    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_in);
    cudaFree(dev_histogramOut);
    cudaFree(dev_lookUp);
    cudaFree(dev_out);
}
extern "C" __declspec(dllexport) void RunAVGKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, int matrixDims) {
    unsigned char* dev_gray; // Eszköz memória a szürkeárnyalatú kép számára
    unsigned char* dev_out;  // Eszköz memória a kimeneti kép számára
    unsigned char* dev_in;   // Eszköz memória a bemeneti kép számára
    int* dev_matrix;         // Eszköz memória a maszkmátrix számára

    // Számítsuk ki a szükséges memória méretét
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    // Maszk méretének meghatározása
    int matrix_size = matrixDims * matrixDims;
    int* matrix = new int[matrix_size]; // Maszk dinamikus létrehozása

    // Maszk inicializálása (egységes értékek)
    for (int i = 0; i < matrix_size; i++) {
        matrix[i] = 1; // Egységértékek
    }

    // Eszközmemória lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_matrix, matrix_size * sizeof(int));

    // Másoljuk a bemeneti képet az eszköz memóriájába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix, matrix, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    // Blokk és rács méretének beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Szürkeárnyalatú konvertálás
    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    // Maszk alkalmazása a szürkeárnyalatú képre
    MaskKernel << <grid, block >> > (dev_gray, dev_out, width, height, widthStep, channels, dev_matrix, matrixDims, matrixDims);
    cudaDeviceSynchronize();

    // Kimeneti kép átmásolása a gazdagép memóriájába
    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_out);
    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_matrix);
}
extern "C" __declspec(dllexport) void RunGaussKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, int matrixDims, double sigma) {
    unsigned char* dev_in; // Eszköz memória a bemeneti kép számára
    unsigned char* dev_gray; // Eszköz memória a szürkeárnyalatú kép számára
    unsigned char* dev_out;  // Eszköz memória a kimeneti kép számára
    int* dev_matrix;         // Eszköz memória a maszkmátrix számára

    // Számítsuk ki a szükséges memória méretét
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    // Maszk méretének meghatározása
    int matrix_size = matrixDims * matrixDims;
    int* matrix = new int[matrix_size]; // Maszk dinamikus létrehozása
    int halfSize = matrixDims / 2; // Maszk fele
    int idx = 0; // Index a maszkhoz
    long sum = 0; // Összeg a maszk normalizálásához

    // Gauss-maszk értékeinek számítása
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            // Gauss-függvény alkalmazása
            double value = (1.0 / (2.0 * 3.141592 * sigma * sigma)) *
                std::exp(-(i * i + j * j) / (2 * sigma * sigma));
            matrix[idx] = value * 10000; // Skálázott maszk
            sum += matrix[idx++]; // Összeg hozzáadása
        }
    }

    // Maszk normalizálása
    for (size_t i = 0; i < matrix_size; i++)
    {
        matrix[i] = matrix[i] / (double)sum * 10000; // Normalizálás
    }

    // Eszközmemória lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_matrix, matrix_size * sizeof(int));

    // Másoljuk a bemeneti képet az eszköz memóriájába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix, matrix, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    // Blokk és rács méretének beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Szürkeárnyalatú konvertálás
    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    // Gauss-maszk alkalmazása a szürkeárnyalatú képre
    MaskKernel << <grid, block >> > (dev_gray, dev_out, width, height, widthStep, channels, dev_matrix, matrixDims, matrixDims);
    cudaDeviceSynchronize();

    // Kimeneti kép átmásolása a gazdagép memóriájába
    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_out);
    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_matrix);
}
extern "C" __declspec(dllexport) void RunSobelKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels)
{
    unsigned char* dev_in;        // Bemeneti kép eszközmemória
    unsigned char* dev_gray;      // Szürkeárnyalatú kép eszközmemória
    unsigned char* dev_gray_blure; // Homályosított szürkeárnyalatú kép
    unsigned char* dev_gray_x;    // Sobel X irányú éltérkép
    unsigned char* dev_gray_y;    // Sobel Y irányú éltérkép
    unsigned char* dev_out;       // Kimeneti kép eszközmemória
    int* dev_matrix_x;           // X irányú Sobel maszk
    int* dev_matrix_y;           // Y irányú Sobel maszk
    int* dev_matrix_blure;       // Homályosító maszk

    // Memória méretek kiszámítása
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    // Sobel maszkok inicializálása
    int matrix_x[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
    int matrix_y[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

    // Eszközmemóriák lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_gray_blure, graySize);
    cudaMalloc((void**)&dev_gray_x, graySize);
    cudaMalloc((void**)&dev_gray_y, graySize);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_matrix_x, 9 * sizeof(int));
    cudaMalloc((void**)&dev_matrix_y, 9 * sizeof(int));

    // Másoljuk a bemeneti képet és a Sobel maszkokat az eszköz memóriába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_x, matrix_x, 9 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_y, matrix_y, 9 * sizeof(int), cudaMemcpyHostToDevice);

    // Blokk és rács méretek beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Szürkeárnyalatú kép konvertálása
    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    // Homályosító maszk előkészítése
    double sigma = 0.5;
    int matrixDimsBlure = 7;
    int matrix_size_blure = matrixDimsBlure * matrixDimsBlure;
    int* matrix = new int[matrix_size_blure];
    int halfSize = matrixDimsBlure / 2;
    long sum = 0;

    // Homályosító maszk létrehozása a Gauss-függvény alapján
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double value = (1.0 / (2.0 * 3.141592 * sigma * sigma)) * std::exp(-(i * i + j * j) / (2 * sigma * sigma));
            matrix[(i + halfSize) * matrixDimsBlure + (j + halfSize)] = value * 10000;
            sum += matrix[(i + halfSize) * matrixDimsBlure + (j + halfSize)];
        }
    }

    // Normalizálás
    for (int i = 0; i < matrix_size_blure; i++) {
        matrix[i] = matrix[i] / (double)sum * 10000;
    }

    // Homályosító maszk átmásolása az eszköz memóriába
    cudaMalloc((void**)&dev_matrix_blure, matrix_size_blure * sizeof(int));
    cudaMemcpy(dev_matrix_blure, matrix, matrix_size_blure * sizeof(int), cudaMemcpyHostToDevice);

    // Homályosítás a szürkeárnyalatú képen
    MaskKernel << <grid, block >> > (dev_gray, dev_gray_blure, width, height, widthStep, channels, dev_matrix_blure, matrixDimsBlure, matrixDimsBlure);
    cudaDeviceSynchronize();

    // Sobel élsimítás
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_gray_x, width, height, widthStep, channels, dev_matrix_x, 3, 3);
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_gray_y, width, height, widthStep, channels, dev_matrix_y, 3, 3);
    cudaDeviceSynchronize();

    // Sobel kernel alkalmazása
    SobelKernel << <grid, block >> > (dev_gray_x, dev_gray_y, dev_out, width, height, widthStep);
    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_out);
    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_gray_x);
    cudaFree(dev_gray_y);
    cudaFree(dev_matrix_x);
    cudaFree(dev_matrix_y);
    cudaFree(dev_matrix_blure);
}
extern "C" __declspec(dllexport) void RunLaplaceKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels)
{
    unsigned char* dev_in;          // Bemeneti kép eszközmemória
    unsigned char* dev_gray;        // Szürkeárnyalatú kép eszközmemória
    unsigned char* dev_gray_blure;  // Homályosított szürkeárnyalatú kép
    unsigned char* dev_out;         // Kimeneti kép eszközmemória
    int* dev_matrix_laplace;       // Laplace maszk
    int* dev_matrix_blure;         // Homályosító maszk

    // Memória méretek kiszámítása
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    // Laplace maszk inicializálása
    int matrix_laplace[9] = { 0, -1, 0,
                               -1, 4, -1,
                               0, -1, 0 };

    // Eszközmemóriák lefoglalása
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_gray_blure, graySize);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_matrix_laplace, 9 * sizeof(int));

    // Másoljuk a bemeneti képet és a Laplace maszkot az eszköz memóriába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_laplace, matrix_laplace, 9 * sizeof(int), cudaMemcpyHostToDevice);

    // Blokk és rács méretek beállítása
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Szürkeárnyalatú kép konvertálása
    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);

    // Homályosító maszk előkészítése
    {
        double sigma = 0.5;
        int matrixDimsBlure = 7;
        int matrix_size_blure = matrixDimsBlure * matrixDimsBlure;
        int* matrix = new int[matrix_size_blure];
        int halfSize = matrixDimsBlure / 2;
        long sum = 0;

        // Homályosító maszk létrehozása
        for (int i = -halfSize; i <= halfSize; i++) {
            for (int j = -halfSize; j <= halfSize; j++) {
                double value = (1.0 / (2.0 * 3.141592 * sigma * sigma)) * std::exp(-(i * i + j * j) / (2 * sigma * sigma));
                matrix[(i + halfSize) * matrixDimsBlure + (j + halfSize)] = value * 10000;
                sum += matrix[(i + halfSize) * matrixDimsBlure + (j + halfSize)];
            }
        }

        // Normalizálás
        for (int i = 0; i < matrix_size_blure; i++) {
            matrix[i] = matrix[i] / (double)sum * 10000;
        }

        // Homályosító maszk átmásolása az eszköz memóriába
        cudaMalloc((void**)&dev_matrix_blure, matrix_size_blure * sizeof(int));
        cudaMemcpy(dev_matrix_blure, matrix, matrix_size_blure * sizeof(int), cudaMemcpyHostToDevice);

        // Homályosítás a szürkeárnyalatú képen
        MaskKernel << <grid, block >> > (dev_gray, dev_gray_blure, width, height, widthStep, channels, dev_matrix_blure, matrixDimsBlure, matrixDimsBlure);
        cudaDeviceSynchronize();
    }

    // Laplace kernel alkalmazása
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_out, width, height, widthStep, channels, dev_matrix_laplace, 3, 3);
    cudaDeviceSynchronize();

    // Kimeneti kép másolása vissza a hosztra
    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_out);
    cudaFree(dev_matrix_laplace);
    cudaFree(dev_matrix_blure);
}
extern "C" __declspec(dllexport) void RunImportantPointKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels)
{
    // Deklaráljuk a szükséges eszközmemóriapuffereket
    unsigned char* dev_in;            // Bemeneti kép
    unsigned char* dev_gray;          // Szürkeárnyalatú kép
    unsigned char* dev_gray_blure;    // Homályosított szürkeárnyalatú kép
    unsigned char* dev_derival_x;     // X irányú első derivált
    unsigned char* dev_derival_y;     // Y irányú első derivált
    unsigned char* dev_derival_xy;    // XY irányú első derivált
    unsigned char* dev_DoubleDerival_x; // X irányú másodrendű derivált
    unsigned char* dev_DoubleDerival_y; // Y irányú másodrendű derivált

    unsigned int* dev_sumderival_xy;  // XY irányú első derivált összegzése
    unsigned int* dev_sumDoubleDerival_x; // X irányú másodrendű derivált összegzése
    unsigned int* dev_sumDoubleDerival_y; // Y irányú másodrendű derivált összegzése

    unsigned char* dev_out;            // Kimeneti kép
    int* dev_matrix_derival_x;        // X irányú derivált maszk
    int* dev_matrix_derival_y;        // Y irányú derivált maszk
    int* dev_matrix_blure;            // Homályosító maszk

    // Különböző memória méretek számítása
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);
    size_t sumderival = (width * height) * sizeof(unsigned int);

    // Maszk méretek beállítása
    int matrixDims = 3;
    int matrix_size = 9;

    // X irányú derivált maszk (Sobel maszk)
    int matrix_derival_x[9] = { 1, 1, 1,
                                0, 0, 0,
                                -1, -1, -1 };

    // Y irányú derivált maszk (Sobel maszk)
    int matrix_derival_y[9] = { 1, 0, -1,
                                1, 0, -1,
                                1, 0, -1 };

    // Eszközmemória lefoglalása a bemeneti és köztes képekhez
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_gray_blure, graySize);
    cudaMalloc((void**)&dev_derival_x, graySize);
    cudaMalloc((void**)&dev_derival_y, graySize);
    cudaMalloc((void**)&dev_derival_xy, graySize);
    cudaMalloc((void**)&dev_DoubleDerival_x, graySize);
    cudaMalloc((void**)&dev_DoubleDerival_y, graySize);

    // Összegzési pufferekhez szükséges memória lefoglalása
    cudaMalloc((void**)&dev_sumderival_xy, sumderival);
    cudaMalloc((void**)&dev_sumDoubleDerival_x, sumderival);
    cudaMalloc((void**)&dev_sumDoubleDerival_y, sumderival);

    // Kimeneti kép puffere
    cudaMalloc((void**)&dev_out, size);

    // Maszkok memóriájának lefoglalása
    cudaMalloc((void**)&dev_matrix_derival_x, matrix_size * sizeof(int));
    cudaMalloc((void**)&dev_matrix_derival_y, matrix_size * sizeof(int));

    // Bemeneti kép másolása az eszköz memóriába
    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    // Derivált maszkok másolása az eszköz memóriába
    cudaMemcpy(dev_matrix_derival_x, matrix_derival_x, matrix_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_derival_y, matrix_derival_y, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    // Képesség és blokk méretek beállítása a CUDA kernel indításához
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    // Szürkeárnyalatú kép generálása
    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);

    {
        // Homályosító maszk generálása Gauss-függvény alapján
        double sigma = 0.5;
        int matrixDimsBlure = 7;
        int matrix_size_blure = matrixDimsBlure * matrixDimsBlure;
        int* matrix = new int[matrix_size_blure]; // Maszk tömb
        int halfSize = matrixDimsBlure / 2; // A maszk fele
        int idx = 0; // Index a maszkhoz
        long sum = 0; // Összeg a maszk értékeinek

        // Gauss-maszk generálása
        for (int i = -halfSize; i <= halfSize; i++) {
            for (int j = -halfSize; j <= halfSize; j++) {
                double value = (1.0 / (2.0 * 3.141592 * sigma * sigma)) *
                    std::exp(-(i * i + j * j) / (2 * sigma * sigma));
                matrix[idx] = value * 10000; // Érték normalizálása
                sum += matrix[idx++]; // Összeg frissítése
            }
        }
        // Maszk normalizálása
        for (size_t i = 0; i < matrix_size_blure; i++)
        {
            matrix[i] = matrix[i] / (double)sum * 10000;
        }

        // Homályosító maszk másolása az eszköz memóriába
        cudaMalloc((void**)&dev_matrix_blure, matrix_size_blure * sizeof(int));
        cudaMemcpy(dev_matrix_blure, matrix, matrix_size_blure * sizeof(int), cudaMemcpyHostToDevice);

        // Homályosító kernel indítása
        MaskKernel << <grid, block >> > (dev_gray, dev_gray_blure, width, height, widthStep, channels, dev_matrix_blure, matrixDimsBlure, matrixDimsBlure);
        cudaDeviceSynchronize(); // Szinkronizálás
    }

    // X és Y irányú deriváltak számítása
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_derival_x, width, height, widthStep, channels, dev_matrix_derival_x, matrixDims, matrixDims);
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_derival_y, width, height, widthStep, channels, dev_matrix_derival_y, matrixDims, matrixDims);
    cudaDeviceSynchronize(); // Szinkronizálás

    // XY irányú és másodrendű deriváltak számítása
    MaskKernel << <grid, block >> > (dev_derival_x, dev_derival_xy, width, height, widthStep, channels, dev_matrix_derival_y, matrixDims, matrixDims);
    MaskKernel << <grid, block >> > (dev_derival_x, dev_DoubleDerival_x, width, height, widthStep, channels, dev_matrix_derival_x, matrixDims, matrixDims);
    MaskKernel << <grid, block >> > (dev_derival_y, dev_DoubleDerival_y, width, height, widthStep, channels, dev_matrix_derival_y, matrixDims, matrixDims);
    cudaDeviceSynchronize(); // Szinkronizálás

    // Összegzés kiszámítása a környezeti területen
    int area = 7;
    SumAreaKernel << <grid, block >> > (dev_derival_xy, dev_sumderival_xy, width, height, widthStep, area);
    SumAreaKernel << <grid, block >> > (dev_DoubleDerival_x, dev_sumDoubleDerival_x, width, height, widthStep, area);
    SumAreaKernel << <grid, block >> > (dev_DoubleDerival_y, dev_sumDoubleDerival_y, width, height, widthStep, area);

    // Fontos pontok azonosítása a kimeneti képben
    double k = 0.25; // Különböző pontok azonosításához használt súly
    int th = 43000; // Küszöbérték a fontos pontok meghatározásához
    KLTKernel << <grid, block >> > (dev_gray, dev_out, dev_sumderival_xy, dev_sumDoubleDerival_x, dev_sumDoubleDerival_y, width, height, widthStep, k, th);
    cudaDeviceSynchronize(); // Szinkronizálás

    // Eredmény másolása a hosztra
    cudaMemcpy(pictureOut, dev_out, size, cudaMemcpyDeviceToHost);

    // Eszközmemória felszabadítása
    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_gray_blure);
    cudaFree(dev_derival_x);
    cudaFree(dev_derival_y);
    cudaFree(dev_derival_xy);
    cudaFree(dev_DoubleDerival_x);
    cudaFree(dev_DoubleDerival_y);
    cudaFree(dev_sumderival_xy);
    cudaFree(dev_sumDoubleDerival_x);
    cudaFree(dev_sumDoubleDerival_y);
    cudaFree(dev_out);
    cudaFree(dev_matrix_blure);
    cudaFree(dev_matrix_derival_x);
    cudaFree(dev_matrix_derival_y);
}

