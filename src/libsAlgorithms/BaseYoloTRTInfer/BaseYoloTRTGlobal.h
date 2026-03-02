

#ifndef BASEYOLOGLOBAL_H
#define BASEYOLOGLOBAL_H

#if defined(_WIN32) || defined(_WIN64)
#ifdef BASEYOLOINFER_EXPORTS
#define BASEYOLOINFER_EXPORT __declspec(dllexport)
#else
#define BASEYOLOINFER_EXPORT __declspec(dllimport)
#endif

#else
#define BASEYOLOINFER_EXPORT
#endif

#endif