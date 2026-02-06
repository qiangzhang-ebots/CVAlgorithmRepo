#pragma once

#if defined(_WIN32) || defined(_WIN64)
	#if defined(ARITHMETICALGORITHMDEMO_BUILD)
		#define ARITHMETICALGORITHMDEMO_EXPORT __declspec(dllexport)
	#else
		#define ARITHMETICALGORITHMDEMO_EXPORT __declspec(dllimport)
	#endif
#else
	#define ARITHMETICALGORITHMDEMO_EXPORT
#endif
