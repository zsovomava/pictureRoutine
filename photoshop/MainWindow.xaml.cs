using Microsoft.Win32;
using ScottPlot;
using System.Collections;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using static System.Net.Mime.MediaTypeNames;

namespace photoshop
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunNegateKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RungammaKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels,
        float gamma
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunLogKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels,
        float C
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunGrayKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunHistogramKernel(
        byte[] pictureIn,
        int[] histogramOut,
        int width,
        int height,
        int widthStep,
        int channels
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunHistogramEqualizationKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunAVGKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels,
        int matrixDims
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunGaussKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels,
        int matrixdims,
        double sigma
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunSobelKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunLaplaceKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels
    );
        [DllImport("pictureRoutine.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void RunImportantPointKernel(
        byte[] pictureIn,
        byte[] pictureOut,
        int width,
        int height,
        int widthStep,
        int channels
    );

        Bitmap picIn;
        Bitmap picOut;
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Tallozas_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif|All Files|*.*"; // Szűrés a képfájlokra

            if (openFileDialog.ShowDialog() == true)
            {
                // Kép betöltése és megjelenítése
                picIn= new Bitmap(openFileDialog.FileName);
                PictureInput.Source = BitmapToImageSource(picIn); // Kép megjelenítése az Image kontrollal
            }

        }
        private void Negativ_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[pictureIn.Length];

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunNegateKernel(pictureIn, pictureOut, width, height, widthStep, channels);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";

            picOut = ByteArrayToBitmap(pictureOut, width, height);
            
            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;
        }
        private void Gamma_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[pictureIn.Length];
            float gamma; 
            var success = float.TryParse(parameter_Number.Text, out gamma);
            if (!success)
            {
                gamma = 1.2f;
            }

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RungammaKernel(pictureIn, pictureOut, width, height, widthStep, channels, gamma);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";

            picOut = ByteArrayToBitmap(pictureOut, width, height);

            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;
        }
        private void log_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[pictureIn.Length];
            float c;
            var success = float.TryParse(parameter_Number.Text, out c);
            if (!success)
            {
                c = 25f;
            }
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunLogKernel(pictureIn, pictureOut, width, height, widthStep, channels, c);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";
            

            picOut = ByteArrayToBitmap(pictureOut, width, height);

            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;
        }
        private void gray_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[(width+widthStep)*height];
            
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunGrayKernel(pictureIn, pictureOut, width, height, widthStep, channels);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";
            
            
            picOut = ByteArrayToBitmapGray(pictureOut, width, height);
            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;
        }
        private void histogram_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            int[] hisztogramOut = new int[256];

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunHistogramKernel(pictureIn, hisztogramOut, width, height, widthStep, channels);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";
            
            ScottPlot.Plot myPlot = new();
            var barPlot = myPlot.Add.Bars(Array.ConvertAll(hisztogramOut, i => (double)i));


            myPlot.Title("Intensity Histogram");
            myPlot.YLabel("Number of Pixels");
            myPlot.XLabel("Intensity");
            myPlot.SavePng("Histogram.png", 1024, 512);
            
            //PictureOutput.Source = BitmapToImageSource(ByteArrayToBitmap(myPlot.GetImageBytes(1024, 512), 1024, 512));
        }
        private void histogram_delay_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[(width + widthStep) * height];

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunHistogramEqualizationKernel(pictureIn, pictureOut, width, height, widthStep, channels);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";
            
            
            picOut = ByteArrayToBitmapGray(pictureOut, width, height);
            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;
        }

        private void AVG_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[(width + widthStep) * height];
            int matrixDims;
            var success = int.TryParse(parameter_Number.Text, out matrixDims);
            if (!success)
            {
                matrixDims = 5;
            }

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunAVGKernel(pictureIn, pictureOut, width, height, widthStep, channels, matrixDims);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";

            picOut = ByteArrayToBitmapGray(pictureOut, width, height);

            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;
        }
        private void Gauss_click(object sender, RoutedEventArgs e)
        {
                int width = picIn.Width;
                int height = picIn.Height;
                int channels = 3;
                int widthStep = (4 - ((width * channels) % 4)) % 4;
                byte[] pictureIn = BitmapToByteArray(picIn);
                byte[] pictureOut = new byte[(width + widthStep) * height];
                int matrixdims = 7;
                double sigma;
                var success = double.TryParse(parameter_Number.Text, out sigma);
                if (!success)
                {
                    sigma = 0.5;
                }
            Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                RunGaussKernel(pictureIn, pictureOut, width, height, widthStep, channels,matrixdims, sigma);
                stopwatch.Stop();
                time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";
                
                picOut = ByteArrayToBitmapGray(pictureOut, width, height);

                PictureOutput.Source = BitmapToImageSource(picOut);
                picIn = picOut;
        }
        private void Sobel_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[(width + widthStep) * height];

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunSobelKernel(pictureIn, pictureOut, width, height, widthStep, channels);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms"; 
            
            picOut = ByteArrayToBitmapGray(pictureOut, width, height);

            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;
        }
        private void Laplace_click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[(width + widthStep) * height];

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunLaplaceKernel(pictureIn, pictureOut, width, height, widthStep, channels);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms"; 
            
            
            picOut = ByteArrayToBitmapGray(pictureOut, width, height);

            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;
        }
        private void ImportantPoint_Click(object sender, RoutedEventArgs e)
        {
            int width = picIn.Width;
            int height = picIn.Height;
            int channels = 3;
            int widthStep = (4 - ((width * channels) % 4)) % 4;
            byte[] pictureIn = BitmapToByteArray(picIn);
            byte[] pictureOut = new byte[pictureIn.Length];

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            RunImportantPointKernel(pictureIn, pictureOut, width, height, widthStep, channels);
            stopwatch.Stop();
            time.Text = stopwatch.ElapsedMilliseconds.ToString() + "ms";
            ;

            picOut = ByteArrayToBitmap(pictureOut, width, height);

            PictureOutput.Source = BitmapToImageSource(picOut);
            picIn = picOut;

        }


        BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
        }

        // Bitmap -> byte[] átalakítás
        static byte[] BitmapToByteArray(Bitmap bmp)
        {
            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            var bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            int length = bmpData.Stride * bmp.Height;
            byte[] data = new byte[length];
            System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, data, 0, length);
            bmp.UnlockBits(bmpData);
            return data;
        }

        // byte[] -> Bitmap átalakítás
        static Bitmap ByteArrayToBitmap(byte[] data, int width, int height)
        {
            Bitmap bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            var bmpData = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            System.Runtime.InteropServices.Marshal.Copy(data, 0, bmpData.Scan0, data.Length);
            bmp.UnlockBits(bmpData);
            return bmp;
        }
        static Bitmap ByteArrayToBitmapGray(byte[] data, int width, int height)
        {
            Bitmap bmp = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            var bmpData = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            System.Runtime.InteropServices.Marshal.Copy(data, 0, bmpData.Scan0, data.Length);
            bmp.UnlockBits(bmpData);
            ColorPalette palette = bmp.Palette;
            for (int i = 0; i < 256; i++)
            {
                palette.Entries[i] = System.Drawing.Color.FromArgb(i, i, i);
            }
            bmp.Palette = palette;

            return bmp;
        }

    }
}