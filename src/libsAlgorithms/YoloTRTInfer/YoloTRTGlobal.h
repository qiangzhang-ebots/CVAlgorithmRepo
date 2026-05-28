

#ifndef YOLOGLOBAL_H
#define YOLOGLOBAL_H

#if defined(_WIN32) || defined(_WIN64)
#ifdef YOLOINFER_EXPORTS
#define YOLOINFER_EXPORT __declspec(dllexport)
#else
#define YOLOINFER_EXPORT __declspec(dllimport)
#endif

#else
#define YOLOINFER_EXPORT
#endif

#endif