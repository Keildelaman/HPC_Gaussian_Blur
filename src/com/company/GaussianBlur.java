package com.company;

import static org.jocl.CL.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

import javax.imageio.ImageIO;

import org.jocl.*;

public class GaussianBlur {


    private static final String KERNEL_SOURCE =
            "__kernel void gaussianBlur(__global const uchar4 *input, __global uchar4 *output, __constant float *filter, int filterSize, int horizontal) {"
                    + "int gid = get_global_id(0);"
                    + "int lid = get_local_id(0);"
                    + "int groupSize = get_local_size(0);"
                    + "int globalSize = get_global_size(0);"
                    + "int oppositeSize = globalSize / groupSize;"
                    + "__local float4 localMem[256];"
                    + ""
                    + "localMem[lid] = convert_float4(input[gid]);"
                    + ""
                    + "barrier(CLK_LOCAL_MEM_FENCE);"
                    + ""
                    + "float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);"
                    + "float filterSum = 0.0f;"
                    + "int filterHalf = filterSize / 2;"
                    + "output[gid] = convert_uchar4_sat_rte((float4)(122.0f, 122.0f, 122.0f, 122.0f));"
                    + "for (int i = -filterHalf; i <= filterHalf; i++) {"
                    + "int index;"
                    + "if (horizontal == 1) {"
                    + "index = clamp(lid + i, 0, groupSize - 1);"
                    + "} else {"
                    + "index = clamp(gid + i * oppositeSize, 0, globalSize - 1);"
                    + "localMem[index] = convert_float4(input[index]);"
                    + "}"
                    + "float filterValue = filter[i + filterHalf];"
                    + "sum += filterValue * localMem[index];"
                    + "filterSum += filterValue;"
                    + "}"
                    + ""
                    + "output[gid] = convert_uchar4_sat_rte(sum/filterSum);"
                    + "}";

    public static void main(String[] args) throws IOException {
        BufferedImage inputImage = ImageIO.read(new File("src/com/company/input.png"));
        int width = inputImage.getWidth();
        int height = inputImage.getHeight();
        int[] inputPixels = inputImage.getRGB(0, 0, width, height, null, 0, width);

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(width * height * 4);
        for (int pixel : inputPixels) {
            inputBuffer.put((byte) ((pixel >> 16) & 0xFF));
            inputBuffer.put((byte) ((pixel >> 8) & 0xFF));
            inputBuffer.put((byte) (pixel & 0xFF));
            inputBuffer.put((byte) ((pixel >> 24) & 0xFF));
        }
        inputBuffer.rewind();

        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[0];

        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, devices.length, devices, null);
        cl_device_id device = devices[0];

        long[] maxWorkGroupSizeArray = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.cl_ulong, Pointer.to(maxWorkGroupSizeArray), null);
        long maxWorkGroupSize = maxWorkGroupSizeArray[0];

        cl_context context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);

        cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, device, null, null);

        cl_mem inputMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, Pointer.to(inputBuffer), null);
        cl_mem outputMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * 4, null, null);

        cl_program program = clCreateProgramWithSource(context, 1, new String[]{KERNEL_SOURCE}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        cl_kernel kernel = clCreateKernel(program, "gaussianBlur", null);

        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inputMem));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outputMem));

        float[] filter = {
                0.000134f, 0.004432f, 0.053991f, 0.241971f, 0.398943f, 0.241971f, 0.053991f, 0.004432f, 0.000134f
        };

        int filterSize = 9;
        cl_mem filterMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filterSize * Sizeof.cl_float, Pointer.to(FloatBuffer.wrap(filter)), null);
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(filterMem));
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{filterSize}));


        long[] globalWorkSize = new long[]{width * height};

        if (width <= maxWorkGroupSize && height <= maxWorkGroupSize) {
            ByteBuffer outputBuffer = ByteBuffer.allocateDirect(width * height * 4);

            // horizontal pass
            long[] localWorkSize = new long[]{width};
            int horizontal = 1;
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{horizontal}));
            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, 0, null, null);

            clEnqueueReadBuffer(commandQueue, outputMem, CL_TRUE, 0, width * height * 4, Pointer.to(outputBuffer), 0, null, null);

            // vertical pass
            localWorkSize = new long[]{height};
            horizontal = 0;
            cl_mem inputMem2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, Pointer.to(outputBuffer), null);
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inputMem2));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{horizontal}));
            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, 0, null, null);

            clEnqueueReadBuffer(commandQueue, outputMem, CL_TRUE, 0, width * height * 4, Pointer.to(outputBuffer), 0, null, null);


            int[] outputPixels = new int[width * height];
            for (int i = 0; i < outputPixels.length; i++) {
                int r = outputBuffer.get() & 0xFF;
                int g = outputBuffer.get() & 0xFF;
                int b = outputBuffer.get() & 0xFF;
                int a = outputBuffer.get() & 0xFF;
                outputPixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
            }

            BufferedImage outputImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            outputImage.setRGB(0, 0, width, height, outputPixels, 0, width);
            ImageIO.write(outputImage, "png", new File("output.png"));

            clReleaseMemObject(inputMem);
            clReleaseMemObject(outputMem);
            clReleaseMemObject(filterMem);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
        } else {
            System.err.println("image size exceeds maximum work group size for this device --> use smaller image");
            System.err.println(maxWorkGroupSize);
        }
    }
}


