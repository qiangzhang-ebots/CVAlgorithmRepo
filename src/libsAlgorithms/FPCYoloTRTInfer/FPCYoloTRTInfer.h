#ifndef FPCYOLOINFER_H
#define FPCYOLOINFER_H

<<<<<<< HEAD
#include "YoloTRTInfer/YoloTRTInfer.h"
=======
#include "BaseYoloTRTInfer/BaseYoloTRTInfer.h"
>>>>>>> 906732a55827364bbdfac4815084379630beebb2
#include "BaseTRTInfer/KeypointObjectDefine.h"
#include "FPCYoloTRTInferGlobal.h"

class FPCYOLOTRTINFER_EXPORT FPCYoloTRTInfer : public YoloTRTInfer {
 public:
  FPCYoloTRTInfer();
  ~FPCYoloTRTInfer();

<<<<<<< HEAD
  // 添加 Predict 方法覆盖
  std::vector<KeypointObjectDescriptor> Predict(const cv::Mat& inputImage);
=======
  virtual void Postprocess() override;
  void PostprocessOneObject(const float* output) override;
>>>>>>> 906732a55827364bbdfac4815084379630beebb2
  std::vector<KeypointObjectDescriptor> GetObjects();

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
<<<<<<< HEAD
  virtual void Postprocess() override;
=======
>>>>>>> 906732a55827364bbdfac4815084379630beebb2
  std::vector<KeypointObjectDescriptor> fpc_zif_objs_, m_valid_objs_;
};

#endif