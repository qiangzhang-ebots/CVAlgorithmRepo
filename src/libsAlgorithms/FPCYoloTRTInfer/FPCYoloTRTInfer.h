
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
  std::vector<YoloKeypointObjectDescriptor> GetObjects() const;

 protected:
  std::vector<YoloKeypointObjectDescriptor> fpc_zif_objs_;
};

#endif