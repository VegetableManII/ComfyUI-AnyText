F:.
│  .gitattributes
│  .gitignore
│  README.md
│  requirements-without-translator.txt
│  requirements.txt
│  __init__.py
│
├─AnyText
│  │  nodes.py
│  │  utils.py
│  │
│  ├─AnyText_scripts
│  │  │  AnyText_bert_tokenizer.py
│  │  │  AnyText_dataset_util.py
│  │  │  AnyText_pipeline.py
│  │  │  AnyText_pipeline_util.py
│  │  │  AnyText_t3_dataset.py
│  │  │
│  │  ├─cldm
│  │  │  │  cldm.py
│  │  │  │  ddim_hacked.py
│  │  │  │  embedding_manager.py
│  │  │  │  hack.py
│  │  │  │  logger.py
│  │  │  │  model.py
│  │  │  │  recognizer.py
│  │  │  │
│  │  │  ├─ocr_recog
│  │  │  │  │  common.py
│  │  │  │  │  en_dict.txt
│  │  │  │  │  ppocr_keys_v1.txt
│  │  │  │  │  RecCTCHead.py
│  │  │  │  │  RecModel.py
│  │  │  │  │  RecMv1_enhance.py
│  │  │  │  │  RecSVTR.py
│  │  │  │  │  RNN.py
│  │  │  │  │
│  │  │  │  └─__pycache__
│  │  │  │          common.cpython-311.pyc
│  │  │  │          RecCTCHead.cpython-311.pyc
│  │  │  │          RecModel.cpython-311.pyc
│  │  │  │          RecMv1_enhance.cpython-311.pyc
│  │  │  │          RecSVTR.cpython-311.pyc
│  │  │  │          RNN.cpython-311.pyc
│  │  │  │
│  │  │  └─__pycache__
│  │  │          cldm.cpython-311.pyc
│  │  │          ddim_hacked.cpython-311.pyc
│  │  │          embedding_manager.cpython-311.pyc
│  │  │          model.cpython-311.pyc
│  │  │          recognizer.cpython-311.pyc
│  │  │
│  │  ├─ldm
│  │  │  │  util.py
│  │  │  │
│  │  │  ├─data
│  │  │  │      util.py
│  │  │  │      __init__.py
│  │  │  │
│  │  │  ├─models
│  │  │  │  │  autoencoder.py
│  │  │  │  │
│  │  │  │  ├─diffusion
│  │  │  │  │  │  ddim.py
│  │  │  │  │  │  ddpm.py
│  │  │  │  │  │  plms.py
│  │  │  │  │  │  recognizer.py
│  │  │  │  │  │  sampling_util.py
│  │  │  │  │  │  __init__.py
│  │  │  │  │  │
│  │  │  │  │  ├─dpm_solver
│  │  │  │  │  │      dpm_solver.py
│  │  │  │  │  │      sampler.py
│  │  │  │  │  │      __init__.py
│  │  │  │  │  │
│  │  │  │  │  ├─ocr_recog
│  │  │  │  │  │  │  common.py
│  │  │  │  │  │  │  en_dict.txt
│  │  │  │  │  │  │  ppocr_keys_v1.txt
│  │  │  │  │  │  │  RecCTCHead.py
│  │  │  │  │  │  │  RecModel.py
│  │  │  │  │  │  │  RecMv1_enhance.py
│  │  │  │  │  │  │  RecSVTR.py
│  │  │  │  │  │  │  RNN.py
│  │  │  │  │  │  │
│  │  │  │  │  │  └─__pycache__
│  │  │  │  │  │          common.cpython-311.pyc
│  │  │  │  │  │          RecCTCHead.cpython-311.pyc
│  │  │  │  │  │          RecModel.cpython-311.pyc
│  │  │  │  │  │          RecMv1_enhance.cpython-311.pyc
│  │  │  │  │  │          RecSVTR.cpython-311.pyc
│  │  │  │  │  │          RNN.cpython-311.pyc
│  │  │  │  │  │
│  │  │  │  │  └─__pycache__
│  │  │  │  │          ddim.cpython-311.pyc
│  │  │  │  │          ddpm.cpython-311.pyc
│  │  │  │  │          recognizer.cpython-311.pyc
│  │  │  │  │          __init__.cpython-311.pyc
│  │  │  │  │
│  │  │  │  └─__pycache__
│  │  │  │          autoencoder.cpython-311.pyc
│  │  │  │
│  │  │  ├─modules
│  │  │  │  │  attention.py
│  │  │  │  │  ema.py
│  │  │  │  │
│  │  │  │  ├─diffusionmodules
│  │  │  │  │  │  model.py
│  │  │  │  │  │  openaimodel.py
│  │  │  │  │  │  upscaling.py
│  │  │  │  │  │  util.py
│  │  │  │  │  │  __init__.py
│  │  │  │  │  │
│  │  │  │  │  └─__pycache__
│  │  │  │  │          model.cpython-311.pyc
│  │  │  │  │          openaimodel.cpython-311.pyc
│  │  │  │  │          util.cpython-311.pyc
│  │  │  │  │          __init__.cpython-311.pyc
│  │  │  │  │
│  │  │  │  ├─distributions
│  │  │  │  │  │  distributions.py
│  │  │  │  │  │  __init__.py
│  │  │  │  │  │
│  │  │  │  │  └─__pycache__
│  │  │  │  │          distributions.cpython-311.pyc
│  │  │  │  │          __init__.cpython-311.pyc
│  │  │  │  │
│  │  │  │  ├─encoders
│  │  │  │  │  │  modules.py
│  │  │  │  │  │  __init__.py
│  │  │  │  │  │
│  │  │  │  │  └─__pycache__
│  │  │  │  │          modules.cpython-311.pyc
│  │  │  │  │          __init__.cpython-311.pyc
│  │  │  │  │
│  │  │  │  ├─image_degradation
│  │  │  │  │  │  bsrgan.py
│  │  │  │  │  │  bsrgan_light.py
│  │  │  │  │  │  utils_image.py
│  │  │  │  │  │  __init__.py
│  │  │  │  │  │
│  │  │  │  │  └─utils
│  │  │  │  │          test.png
│  │  │  │  │
│  │  │  │  ├─midas
│  │  │  │  │  │  api.py
│  │  │  │  │  │  utils.py
│  │  │  │  │  │  __init__.py
│  │  │  │  │  │
│  │  │  │  │  └─midas
│  │  │  │  │          base_model.py
│  │  │  │  │          blocks.py
│  │  │  │  │          dpt_depth.py
│  │  │  │  │          midas_net.py
│  │  │  │  │          midas_net_custom.py
│  │  │  │  │          transforms.py
│  │  │  │  │          vit.py
│  │  │  │  │          __init__.py
│  │  │  │  │
│  │  │  │  └─__pycache__
│  │  │  │          attention.cpython-311.pyc
│  │  │  │          ema.cpython-311.pyc
│  │  │  │
│  │  │  └─__pycache__
│  │  │          util.cpython-311.pyc
│  │  │
│  │  └─__pycache__
│  │          AnyText_bert_tokenizer.cpython-311.pyc
│  │          AnyText_dataset_util.cpython-311.pyc
│  │          AnyText_pipeline.cpython-311.pyc
│  │          AnyText_pipeline_util.cpython-311.pyc
│  │          AnyText_t3_dataset.cpython-311.pyc
│  │
│  ├─assets
│  │      AnyText-wf.png
│  │      clip_model.jpg
│  │      README-Zh-CN.md
│  │      zh2en_model.jpg
│  │
│  ├─example_images
│  │      edit12.png
│  │      edit13.png
│  │      edit15.png
│  │      edit16.png
│  │      edit2.png
│  │      edit3.png
│  │      edit5.png
│  │      ref12.png
│  │      ref13.jpg
│  │      ref15.jpeg
│  │      ref16.jpeg
│  │      ref2.jpg
│  │      ref3.jpg
│  │      ref5.jpg
│  │
│  ├─models_yaml
│  │      anytext_sd15.yaml
│  │
│  ├─ocr_weights
│  │      ppocr_keys_v1.txt
│  │      ppv3_rec.pth
│  │
│  ├─temp_dir
│  │      AnyText_manual_mask_pos_img.png
│  │      AnyText_random_mask_pos_img.png
│  │      AnyText_temp.txt
│  │
│  └─__pycache__
│          nodes.cpython-311.pyc
│          utils.cpython-311.pyc
│
└─__pycache__
        __init__.cpython-311.pyc