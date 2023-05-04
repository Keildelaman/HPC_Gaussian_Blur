package com.company;
import org.jocl.*;
import static org.jocl.CL.*;

public class MyOpenCLProgram {

    public static void main(String[] args) {
        // Initialize OpenCL
        setExceptionsEnabled(true);
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(1, platforms, null);
        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, devices, null);
        cl_context context = CL.clCreateContext(null, 1, devices, null, null, null);

        // Use the context to execute OpenCL code
        // ...

        // Clean up resources when finished
        clReleaseContext(context);


    }
}
