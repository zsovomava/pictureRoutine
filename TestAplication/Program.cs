using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

class Program
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
    static void Main(string[] args)
    {
        Bitmap bmp = new Bitmap("test.png");
        int width = bmp.Width;
        int height = bmp.Height;
        int channels = 3;
        int widthStep = (4-((width * channels)%4))%4;
        byte[] pictureIn = BitmapToByteArray(bmp);
        byte[] pictureOut = new byte[pictureIn.Length];

        RunNegateKernel(pictureIn, pictureOut, width, height, widthStep, channels);

        Bitmap result = ByteArrayToBitmap(pictureOut, width, height);
        result.Save("output_image.png", ImageFormat.Png);

        Console.WriteLine("Kép negálása sikeresen lefutott!");


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
}
