// DllExample.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "DllExample.h"


// This is an example of an exported variable
DLLEXAMPLE_API int nDllExample=0;

// This is an example of an exported function.
DLLEXAMPLE_API int fnDllExample(void)
{
    return 0;
}

// This is the constructor of a class that has been exported.
CDllExample::CDllExample()
{
    return;
}
