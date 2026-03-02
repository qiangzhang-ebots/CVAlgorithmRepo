

#ifndef FPCYOLOGLOBAL_H
#define FPCYOLOGLOBAL_H

#if defined(_WIN32) || defined(_WIN64)
#ifdef FPCYOLOTRTINFER_EXPORTS
#define FPCYOLOTRTINFER_EXPORT __declspec(dllexport)
#else
#define FPCYOLOTRTINFER_EXPORT __declspec(dllimport)
#endif

#else
#define FPCYOLOTRTINFER_EXPORT
#endif

#endif