# MedAI
For the hackathon in 04/10/2026


Group A: CS Hardcore Development Team (3 members)

Member 1 (Data & Pipeline):
Responsible for data loading, preprocessing, and validation split. Familiar with medical imaging libraries such as MONAI or NiBabel, handling 3D NIfTI/DICOM formats, and performing intensity normalization.

Member 2 (Model & Architecture):
Responsible for building the model architecture that includes image feature extraction + tracer fusion, and implementing the training logic (training loop).

Member 3 (Ops & Tuning):
Responsible for writing inference/testing code, generating submission files, monitoring training with WandB/TensorBoard, and later handling hyperparameter tuning and model ensembling.

Group B: Biomedical / Neuroscience Team (2 members)

Task 1: Domain Knowledge Expert (Knowledge Advantage):

Use off-the-shelf visualization tools (such as ITK-SNAP or 3D Slicer) to visually inspect PET images with different tracers, and summarize noise patterns and brightness differences.
Review literature: Where does Amyloid primarily accumulate (e.g., cortex)? Which regions correspond to background noise (e.g., white matter)?
Feed this information back to the CS team to guide image cropping and attention mechanisms.

Task 2: Error Analysis:
After the CS team produces initial results, take the samples with the largest prediction errors and have the BME team analyze them. They can determine whether errors are due to imaging artifacts, severe atrophy, or tracer-specific issues, and provide guidance for further optimization.

Task 3: Storyline & Presentation Preparation:
Organize the rationale behind your chosen network architecture and how tracer differences are handled. Prepare a publication-level Methodology section and presentation slides in advance.
