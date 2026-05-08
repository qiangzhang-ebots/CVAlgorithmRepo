

#ifndef HRNETINFERGLOBAL_H
#define HRNETINFERGLOBAL_H

#if defined(_WIN32) || defined(_WIN64)
#ifdef HRNETINFER_EXPORTS
#define HRNETINFER_EXPORT __declspec(dllexport)
#else
#define HRNETINFER_EXPORT __declspec(dllimport)
#endif

#else
#define HRNETINFER_EXPORT
#endif

#endif