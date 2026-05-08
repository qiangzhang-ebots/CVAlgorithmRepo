

#ifndef BASETRTGLOBAL_H
#define BASETRTGLOBAL_H

#if defined(_WIN32) || defined(_WIN64)
#ifdef BASETRTINFER_EXPORTS
#define BASETRTINFER_EXPORT __declspec(dllexport)
#else
#define BASETRTINFER_EXPORT __declspec(dllimport)
#endif

#else
#define BASETRTINFER_EXPORT
#endif

#endif