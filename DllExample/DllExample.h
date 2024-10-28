// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the DLLEXAMPLE_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// DLLEXAMPLE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef DLLEXAMPLE_EXPORTS
#define DLLEXAMPLE_API __declspec(dllexport)
#else
#define DLLEXAMPLE_API __declspec(dllimport)
#endif

// This class is exported from the dll
class DLLEXAMPLE_API CDllExample {
public:
	CDllExample(void);
	// TODO: add your methods here.
};

extern DLLEXAMPLE_API int nDllExample;

DLLEXAMPLE_API int fnDllExample(void);
