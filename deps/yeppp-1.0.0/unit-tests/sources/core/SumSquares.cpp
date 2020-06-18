/*
 *                       Yeppp! library implementation
 *                   This file is auto-generated by Peach-Py,
 *        Portable Efficient Assembly Code-generator in Higher-level Python,
 *                  part of the Yeppp! library infrastructure
 * This file is part of Yeppp! library and licensed under the New BSD license.
 * See LICENSE.txt for the full text of the license.
 */

#include <yepPredefines.h>
#include <yepPrivate.h>
#include <yepLibrary.h>
#include <library/functions.h>
#include <yepRandom.h>
#include <core/functions.h>
#include <yepBuiltin.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#ifdef YEP_WINDOWS_OS
	#include <windows.h>
	#define YEP_ESCAPE_NORMAL_COLOR ""
	#define YEP_ESCAPE_RED_COLOR ""
	#define YEP_ESCAPE_GREEN_COLOR ""
	#define YEP_ESCAPE_YELLOW_COLOR ""
#else
	#define YEP_ESCAPE_NORMAL_COLOR "\x1B[0m"
	#define YEP_ESCAPE_RED_COLOR "\x1B[31m"
	#define YEP_ESCAPE_GREEN_COLOR "\x1B[32m"
	#define YEP_ESCAPE_YELLOW_COLOR "\x1B[33m"
#endif

static const char* getMicroarchitectureName(YepCpuMicroarchitecture microarchitecture) {
	const YepSize bufferSize = 1024;
	static char buffer[bufferSize];
	YepSize bufferLength = bufferSize - 1;
	YepStatus status = yepLibrary_GetString(YepEnumerationCpuMicroarchitecture, microarchitecture, YepStringTypeDescription, buffer, &bufferLength);
	assert(status == YepStatusOk);
	buffer[bufferLength] = '\0';
	return buffer;
}

static void reportFailedTest(const char* functionName, YepCpuMicroarchitecture microarchitecture) {
	#ifdef YEP_WINDOWS_OS
		CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
		::GetConsoleScreenBufferInfo(::GetStdHandle(STD_OUTPUT_HANDLE), &bufferInfo);
		printf("%s (%s): ", functionName, getMicroarchitectureName(microarchitecture));
		fflush(stdout);
		::SetConsoleTextAttribute(::GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_INTENSITY);
		printf("FAILED\n");
		fflush(stdout);
		::SetConsoleTextAttribute(::GetStdHandle(STD_OUTPUT_HANDLE), bufferInfo.wAttributes);
	#else
		printf("%s (%s): %sFAILED%s\n", functionName, getMicroarchitectureName(microarchitecture), YEP_ESCAPE_RED_COLOR, YEP_ESCAPE_NORMAL_COLOR);
	#endif
}

static void reportFailedTest(const char* functionName, YepCpuMicroarchitecture microarchitecture, float ulpError) {
	#ifdef YEP_WINDOWS_OS
		CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
		::GetConsoleScreenBufferInfo(::GetStdHandle(STD_OUTPUT_HANDLE), &bufferInfo);
		printf("%s (%s): ", functionName, getMicroarchitectureName(microarchitecture));
		fflush(stdout);
		::SetConsoleTextAttribute(::GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_INTENSITY);
		printf("FAILED");
		fflush(stdout);
		::SetConsoleTextAttribute(::GetStdHandle(STD_OUTPUT_HANDLE), bufferInfo.wAttributes);
		printf(" (%f ULP)\n", ulpError);
	#else
		printf("%s (%s): %sFAILED%s (%f ULP)\n", functionName, getMicroarchitectureName(microarchitecture), YEP_ESCAPE_RED_COLOR, YEP_ESCAPE_NORMAL_COLOR, ulpError);
	#endif
}

static void reportPassedTest(const char* functionName, YepCpuMicroarchitecture microarchitecture) {
	#ifdef YEP_WINDOWS_OS
		CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
		::GetConsoleScreenBufferInfo(::GetStdHandle(STD_OUTPUT_HANDLE), &bufferInfo);
		printf("%s (%s): ", functionName, getMicroarchitectureName(microarchitecture));
		fflush(stdout);
		::SetConsoleTextAttribute(::GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		printf("PASSED\n");
		fflush(stdout);
		::SetConsoleTextAttribute(::GetStdHandle(STD_OUTPUT_HANDLE), bufferInfo.wAttributes);
	#else
		printf("%s (%s): %sPASSED%s\n", functionName, getMicroarchitectureName(microarchitecture), YEP_ESCAPE_GREEN_COLOR, YEP_ESCAPE_NORMAL_COLOR);
	#endif
}

static void reportSkippedTest(const char* functionName, YepCpuMicroarchitecture microarchitecture) {
	#ifdef YEP_WINDOWS_OS
		CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
		::GetConsoleScreenBufferInfo(::GetStdHandle(STD_OUTPUT_HANDLE), &bufferInfo);
		printf("%s (%s): ", functionName, getMicroarchitectureName(microarchitecture));
		fflush(stdout);
		::SetConsoleTextAttribute(::GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		printf("SKIPPED\n");
		fflush(stdout);
		::SetConsoleTextAttribute(::GetStdHandle(STD_OUTPUT_HANDLE), bufferInfo.wAttributes);
	#else
		printf("%s (%s): %sSKIPPED%s\n", functionName, getMicroarchitectureName(microarchitecture), YEP_ESCAPE_YELLOW_COLOR, YEP_ESCAPE_NORMAL_COLOR);
	#endif
}

static Yep32s Test_SumSquares_V32f_S32f(Yep64u supportedIsaFeatures, Yep64u supportedSimdFeatures, Yep64u supportedSystemFeatures) {
	YepRandom_WELL1024a rng;
	YepStatus status = yepRandom_WELL1024a_Init(&rng);
	assert(status == YepStatusOk);

	typedef YepStatus (YEPABI* FunctionPointer)(const Yep32f *YEP_RESTRICT, Yep32f *YEP_RESTRICT, YepSize);
	typedef const FunctionDescriptor<FunctionPointer>* DescriptorPointer;
	const DescriptorPointer defaultDescriptor = findDefaultDescriptor(_dispatchTable_yepCore_SumSquares_V32f_S32f);
	const FunctionPointer defaultImplementation = defaultDescriptor->function;
	Yep32s failedTests = 0;

	YEP_ALIGN(64) Yep32f vArray[1088 + (64 / sizeof(Yep32f))];
	Yep32f sumSquares;
	Yep32f sumSquaresInit;
	Yep32f sumSquaresRef;

	status = yepRandom_WELL1024a_GenerateUniform_S32fS32f_V32f_Acc32(&rng, -1.0f, 1.0f, vArray, YEP_COUNT_OF(vArray));
	assert(status == YepStatusOk);
	status = yepRandom_WELL1024a_GenerateUniform_S32fS32f_V32f_Acc32(&rng, -1.0f, 1.0f, &sumSquaresInit, 1);
	assert(status == YepStatusOk);

	for (DescriptorPointer descriptor = &_dispatchTable_yepCore_SumSquares_V32f_S32f[0]; descriptor != defaultDescriptor; descriptor++) {
		const Yep64u unsupportedRequiredFeatures = (descriptor->isaFeatures & ~supportedIsaFeatures) |
			(descriptor->simdFeatures & ~supportedSimdFeatures) |
			(descriptor->systemFeatures & ~supportedSystemFeatures);
		if (unsupportedRequiredFeatures == 0) {
			for (YepSize vOffset = 0; vOffset < 64 / sizeof(Yep32f); vOffset++) {
				for (YepSize length = 0; length < 64; length += 1) {
					sumSquaresRef = sumSquaresInit;
					status = defaultImplementation(&vArray[vOffset], &sumSquaresRef, length);
					assert(status == YepStatusOk);

					sumSquares = sumSquaresInit;
					status = descriptor->function(&vArray[vOffset], &sumSquares, length);
					assert(status == YepStatusOk);

					const Yep32f ulpError = yepBuiltin_Abs_32f_32f(sumSquaresRef - sumSquares) / yepBuiltin_Ulp_32f_32f(sumSquaresRef);
					if (ulpError > 1000.0f) {
						failedTests += 1;
						reportFailedTest("yepCore_SumSquares_V32f_S32f", descriptor->microarchitecture, float(ulpError));
						goto next_descriptor;
					}
				}
				for (YepSize length = 1024; length < 1088; length += 1) {
					sumSquaresRef = sumSquaresInit;
					status = defaultImplementation(&vArray[vOffset], &sumSquaresRef, length);
					assert(status == YepStatusOk);

					sumSquares = sumSquaresInit;
					status = descriptor->function(&vArray[vOffset], &sumSquares, length);
					assert(status == YepStatusOk);

					const Yep32f ulpError = yepBuiltin_Abs_32f_32f(sumSquaresRef - sumSquares) / yepBuiltin_Ulp_32f_32f(sumSquaresRef);
					if (ulpError > 1000.0f) {
						failedTests += 1;
						reportFailedTest("yepCore_SumSquares_V32f_S32f", descriptor->microarchitecture, float(ulpError));
						goto next_descriptor;
					}
				}
			}
			reportPassedTest("yepCore_SumSquares_V32f_S32f", descriptor->microarchitecture);
		} else {
			reportSkippedTest("yepCore_SumSquares_V32f_S32f", descriptor->microarchitecture);
		}
next_descriptor:
		continue;
	}
	return -failedTests;
}

static Yep32s Test_SumSquares_V64f_S64f(Yep64u supportedIsaFeatures, Yep64u supportedSimdFeatures, Yep64u supportedSystemFeatures) {
	YepRandom_WELL1024a rng;
	YepStatus status = yepRandom_WELL1024a_Init(&rng);
	assert(status == YepStatusOk);

	typedef YepStatus (YEPABI* FunctionPointer)(const Yep64f *YEP_RESTRICT, Yep64f *YEP_RESTRICT, YepSize);
	typedef const FunctionDescriptor<FunctionPointer>* DescriptorPointer;
	const DescriptorPointer defaultDescriptor = findDefaultDescriptor(_dispatchTable_yepCore_SumSquares_V64f_S64f);
	const FunctionPointer defaultImplementation = defaultDescriptor->function;
	Yep32s failedTests = 0;

	YEP_ALIGN(64) Yep64f vArray[1088 + (64 / sizeof(Yep64f))];
	Yep64f sumSquares;
	Yep64f sumSquaresInit;
	Yep64f sumSquaresRef;

	status = yepRandom_WELL1024a_GenerateUniform_S64fS64f_V64f_Acc64(&rng, -1.0, 1.0, vArray, YEP_COUNT_OF(vArray));
	assert(status == YepStatusOk);
	status = yepRandom_WELL1024a_GenerateUniform_S64fS64f_V64f_Acc64(&rng, -1.0, 1.0, &sumSquaresInit, 1);
	assert(status == YepStatusOk);

	for (DescriptorPointer descriptor = &_dispatchTable_yepCore_SumSquares_V64f_S64f[0]; descriptor != defaultDescriptor; descriptor++) {
		const Yep64u unsupportedRequiredFeatures = (descriptor->isaFeatures & ~supportedIsaFeatures) |
			(descriptor->simdFeatures & ~supportedSimdFeatures) |
			(descriptor->systemFeatures & ~supportedSystemFeatures);
		if (unsupportedRequiredFeatures == 0) {
			for (YepSize vOffset = 0; vOffset < 64 / sizeof(Yep64f); vOffset++) {
				for (YepSize length = 0; length < 64; length += 1) {
					sumSquaresRef = sumSquaresInit;
					status = defaultImplementation(&vArray[vOffset], &sumSquaresRef, length);
					assert(status == YepStatusOk);

					sumSquares = sumSquaresInit;
					status = descriptor->function(&vArray[vOffset], &sumSquares, length);
					assert(status == YepStatusOk);

					const Yep64f ulpError = yepBuiltin_Abs_64f_64f(sumSquaresRef - sumSquares) / yepBuiltin_Ulp_64f_64f(sumSquaresRef);
					if (ulpError > 1000.0f) {
						failedTests += 1;
						reportFailedTest("yepCore_SumSquares_V64f_S64f", descriptor->microarchitecture, float(ulpError));
						goto next_descriptor;
					}
				}
				for (YepSize length = 1024; length < 1088; length += 1) {
					sumSquaresRef = sumSquaresInit;
					status = defaultImplementation(&vArray[vOffset], &sumSquaresRef, length);
					assert(status == YepStatusOk);

					sumSquares = sumSquaresInit;
					status = descriptor->function(&vArray[vOffset], &sumSquares, length);
					assert(status == YepStatusOk);

					const Yep64f ulpError = yepBuiltin_Abs_64f_64f(sumSquaresRef - sumSquares) / yepBuiltin_Ulp_64f_64f(sumSquaresRef);
					if (ulpError > 1000.0f) {
						failedTests += 1;
						reportFailedTest("yepCore_SumSquares_V64f_S64f", descriptor->microarchitecture, float(ulpError));
						goto next_descriptor;
					}
				}
			}
			reportPassedTest("yepCore_SumSquares_V64f_S64f", descriptor->microarchitecture);
		} else {
			reportSkippedTest("yepCore_SumSquares_V64f_S64f", descriptor->microarchitecture);
		}
next_descriptor:
		continue;
	}
	return -failedTests;
}

int main(int argc, char** argv) {
	YepBoolean testSumSquares_V32f_S32f = YepBooleanFalse;
	YepBoolean testSumSquares_V64f_S64f = YepBooleanFalse;
	if (argc == 1) {
		/* No tests specified: run all tests*/
		testSumSquares_V32f_S32f = YepBooleanTrue;
		testSumSquares_V64f_S64f = YepBooleanTrue;
	} else {
		/* Some tests specified: run only specified tests*/
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "V32f_S32f") == 0) {
				testSumSquares_V32f_S32f = YepBooleanTrue;
			} else if (strcmp(argv[i], "V64f_S64f") == 0) {
				testSumSquares_V64f_S64f = YepBooleanTrue;
			} else {
				fprintf(stderr, "Unknown function name \"%s\"", argv[i]);
				return 1;
			}
		}
	}
	YepStatus status = _yepLibrary_InitCpuInfo();
	assert(status == YepStatusOk);

	Yep64u supportedIsaFeatures, supportedSimdFeatures, supportedSystemFeatures;
	status = yepLibrary_GetCpuIsaFeatures(&supportedIsaFeatures);
	assert(status == YepStatusOk);
	status = yepLibrary_GetCpuSimdFeatures(&supportedSimdFeatures);
	assert(status == YepStatusOk);
	status = yepLibrary_GetCpuSystemFeatures(&supportedSystemFeatures);
	assert(status == YepStatusOk);

	Yep32s failedTests = 0;
	if YEP_LIKELY(testSumSquares_V32f_S32f)
		failedTests += Test_SumSquares_V32f_S32f(supportedIsaFeatures, supportedSimdFeatures, supportedSystemFeatures);
	if YEP_LIKELY(testSumSquares_V64f_S64f)
		failedTests += Test_SumSquares_V64f_S64f(supportedIsaFeatures, supportedSimdFeatures, supportedSystemFeatures);
	return failedTests;
}
