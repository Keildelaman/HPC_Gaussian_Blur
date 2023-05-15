package com.company;

import static org.jocl.CL.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import javax.imageio.ImageIO;

import org.jocl.*;

public class GaussianBlur {


    private static final String KERNEL_SOURCE =
            "__kernel void gaussianBlur(__global const uchar4 *input, __global uchar4 *output, int width, int height, __constant float *filter, int filterSize) {"
                    + "int gidX = get_global_id(0);"
                    + "int gidY = get_global_id(1);"
                    + "if (gidX < width && gidY < height) {"
                    + "float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);"
                    + "float filterSum = 0.0f;"
                    + "int filterHalf = filterSize / 2;"
                    + "for (int y = -filterHalf; y <= filterHalf; y++) {"
                    + "for (int x = -filterHalf; x <= filterHalf; x++) {"
                    + "int clampedX = clamp(gidX + x, 0, width - 1);"
                    + "int clampedY = clamp(gidY + y, 0, height - 1);"
                    + "int index = clampedY * width + clampedX;"
                    + "float filterValue = filter[(y + filterHalf) * filterSize + (x + filterHalf)];"
                    + "sum += filterValue * convert_float4(input[index]);"
                    + "filterSum += filterValue;"
                    + "}"
                    + "}"
                    + "output[gidY * width + gidX] = convert_uchar4_sat_rte(sum / filterSum);"
                    + "}"
                    + "}";

    public static void main(String[] args) throws IOException {
        // Load image
        BufferedImage inputImage = ImageIO.read(new File("src/com/company/input.png"));
        int width = inputImage.getWidth();
        int height = inputImage.getHeight();
        int[] inputPixels = inputImage.getRGB(0, 0, width, height, null, 0, width);

        // Convert image to uchar4
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(width * height * 4);
        for (int pixel : inputPixels) {
            inputBuffer.put((byte) ((pixel >> 16) & 0xFF));
            inputBuffer.put((byte) ((pixel >> 8) & 0xFF));
            inputBuffer.put((byte) (pixel & 0xFF));
            inputBuffer.put((byte) ((pixel >> 24) & 0xFF));
        }
        inputBuffer.rewind();

        // Setup OpenCL
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[0];

        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, devices.length, devices, null);
        cl_device_id device = devices[0];

        // Check the maximum work group size for the device
        long[] maxWorkGroupSizeArray = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.cl_ulong, Pointer.to(maxWorkGroupSizeArray), null);
        long maxWorkGroupSize = maxWorkGroupSizeArray[0];

        cl_context context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);

        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

        cl_mem inputMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, Pointer.to(inputBuffer), null);
        cl_mem outputMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * 4, null, null);

        cl_program program = clCreateProgramWithSource(context, 1, new String[]{KERNEL_SOURCE}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        cl_kernel kernel = clCreateKernel(program, "gaussianBlur", null);

        // Set kernel arguments
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inputMem));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outputMem));
        clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{width}));
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{height}));

        // Gaussian filter
        float[] filter = {
                0.000000f, 0.000001f, 0.000007f, 0.000032f, 0.000053f, 0.000032f, 0.000007f, 0.000001f, 0.000000f,
                0.000001f, 0.000020f, 0.000239f, 0.001072f, 0.001768f, 0.001072f, 0.000239f, 0.000020f, 0.000001f,
                0.000007f, 0.000239f, 0.002915f, 0.013064f, 0.021539f, 0.013064f, 0.002915f, 0.000239f, 0.000007f,
                0.000032f, 0.001072f, 0.013064f, 0.058550f, 0.096533f, 0.058550f, 0.013064f, 0.001072f, 0.000032f,
                0.000053f, 0.001768f, 0.021539f, 0.096533f, 0.159156f, 0.096533f, 0.021539f, 0.001768f, 0.000053f,
                0.000032f, 0.001072f, 0.013064f, 0.058550f, 0.096533f, 0.058550f, 0.013064f, 0.001072f, 0.000032f,
                0.000007f, 0.000239f, 0.002915f, 0.013064f, 0.021539f, 0.013064f, 0.002915f, 0.000239f, 0.000007f,
                0.000001f, 0.000020f, 0.000239f, 0.001072f, 0.001768f, 0.001072f, 0.000239f, 0.000020f, 0.000001f,
                0.000000f, 0.000001f, 0.000007f, 0.000032f, 0.000053f, 0.000032f, 0.000007f, 0.000001f, 0.000000f
        };

        int filterSize = 3;
        cl_mem filterMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filterSize * filterSize * Sizeof.cl_float, Pointer.to(FloatBuffer.wrap(filter)), null);
        clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(filterMem));
        clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{filterSize}));

        long[] localWorkSize = new long[]{16, 16}; // Change this to desired local work group size

        // Make sure the NDRange is valid for the system
        if (localWorkSize[0] * localWorkSize[1] <= maxWorkGroupSize) {
            // Define the local work group size


            // Calculate the global work size as a multiple of the local work size
            long[] globalWorkSize = new long[]{
                    (long) Math.ceil(width / (double) localWorkSize[0]) * localWorkSize[0],
                    (long) Math.ceil(height / (double) localWorkSize[1]) * localWorkSize[1]
            };

            // Execute the kernel
            clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, globalWorkSize, localWorkSize, 0, null, null);

            // Read the output data
            ByteBuffer outputBuffer = ByteBuffer.allocateDirect(width * height * 4);
            clEnqueueReadBuffer(commandQueue, outputMem, CL_TRUE, 0, width * height * 4, Pointer.to(outputBuffer), 0, null, null);

            // Convert uchar4 to int[]
            int[] outputPixels = new int[width * height];
            for (int i = 0; i < outputPixels.length; i++) {
                int r = outputBuffer.get() & 0xFF;
                int g = outputBuffer.get() & 0xFF;
                int b = outputBuffer.get() & 0xFF;
                int a = outputBuffer.get() & 0xFF;
                outputPixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
            }

            // Save the output image
            BufferedImage outputImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            outputImage.setRGB(0, 0, width, height, outputPixels, 0, width);
            ImageIO.write(outputImage, "png", new File("output.png"));

            // Clean up
            clReleaseMemObject(inputMem);
            clReleaseMemObject(outputMem);
            clReleaseMemObject(filterMem);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
        } else {
            System.err.println("The image size exceeds the maximum work group size for the device. Please use a smaller image.");
            System.err.println(maxWorkGroupSize);
        }
    }
}


