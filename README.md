# SIHNet

Deep Learning CT Analysis for Identifying Spontaneous Intracranial Hypotension with Subdural Hematoma

Abstract
Background
Subdural hematoma (SDH) secondary to spontaneous intracranial hypotension (SIH) is often challenging to differentiate from traumatic chronic subdural hematoma (CSDH), particularly if the orthostatic nature of the headache is not recognized. Cranial computed tomography (CT) is widely used as the initial imaging modality; however, SIH-induced SDHs and traumatic CSDHs frequently share similar CT characteristics. An automated method to identify SDH associated with SIH on CT scans could provide substantial clinical value.
Methods
We developed a lightweight 3D convolutional neural network, SIHNet, for automatic identification of SIH-induced SDH on CT images. We trained and internally validated SIHNet on 3D cranial CT scans from 212 SDH patients at Sir Run Run Shaw Hospital (Zhejiang University School of Medicine), including 146 SIH-induced SDH cases and 66 traumatic CSDH cases. The model’s generalizability was further evaluated on an independent external validation set of 18 confirmed SIH-induced SDH cases from the Second Affiliated Hospital of Zhejiang University School of Medicine.
Results
SIHNet achieved an identification accuracy of 88.71% on the internal validation set, outperforming all baseline deep learning models and even experienced neurosurgeons in a simulated diagnostic test. Gradient-weighted Class Activation Mapping (Grad-CAM) visualizations confirmed that SIHNet focused on image regions consistent with known SIH-induced SDH radiographic features. On the external validation set, SIHNet maintained a high accuracy of 83.33%, demonstrating robust performance.
Conclusion
SIHNet showed excellent performance in automatically identifying SIH-induced SDH on CT images. With its strong interpretability, SIHNet may serve as a clinical decision support tool.
Keywords: Spontaneous intracranial hypotension, Subdural hematoma, CT diagnosis, Deep learning, Convolutional neural network

Our implementation is based on the publicly available MedicalNet framework (https://github.com/Tencent/MedicalNet), with task-specific modifications to adapt the model for SIH-related SDH classification.

1. Extract the compressed files:
   - `feature_case/layer4_feature.rar`
   - `feature_case/layer4_output.rar`

2. Run the inference script:
   ```bash
   python infer_from_saved_feature.py
