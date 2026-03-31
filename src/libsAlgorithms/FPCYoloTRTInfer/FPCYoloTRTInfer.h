
#ifndef FPCYOLOINFER_H
#define FPCYOLOINFER_H

#include "BaseYoloTRTInfer/BaseYoloTRTInfer.h"
#include "BaseYoloTRTInfer/YoloObjectDefine.h"
#include "FPCYoloTRTInferGlobal.h"
class FPCYOLOTRTINFER_EXPORT FPCYoloTRTInfer : public BaseYoloTRTInfer {
 public:
  FPCYoloTRTInfer();
  ~FPCYoloTRTInfer();

  virtual void Postprocess() override;
  void PostprocessOneObject(const float* output) override;
  std::vector<YoloKeypointObjectDescriptor> GetObjects();

  /*
    * @brief calculate the overlap between FPC and ZIF
    *  the formula is: overlap = | FPC n ZIF | / FPC
    *  it would return a pair of double, the first one is for the bigger one, and the second one is for the smaller one
    * 
    * Please note that: if the overlap is 0, it means that there is no overlap or failed to detect the object
    *
  */
  std::pair<double, double> CalOverLap();
  std::pair<double, double> CalFpcArea();
 protected:
  std::vector<YoloKeypointObjectDescriptor> fpc_zif_objs_, m_valid_objs_;
};

#endif
