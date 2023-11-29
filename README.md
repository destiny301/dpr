# Patch-based Selection and Refinement for Early Object Detection [WACV2024](https://arxiv.org/abs/2311.02274)

![image](https://github.com/destiny301/dpr/blob/main/flowchart.png)

# Updates
<img src="https://github.com/destiny301/dpr/blob/main/ps_module.png" width="400">
<!-- ![image](https://github.com/destiny301/dpr/blob/main/ps_module.png | width=100) -->
Patch-Selector module code is releaseed.

For Patch-Refinement module, please refer to [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).

# Data
Prepare the data as the following structure:
```shell
root/
├──images/
│  ├── train/
│  │   ├── 000001.jpg
│  │   ├── 000002.jpg
│  │   ├── ......
│  ├── ......
├──masks/
│  ├── val/
│  │   ├── 000001.png
│  │   ├── 000002.png
│  │   ├── ......
```

# Citation
If you use DPR in your research or wish to refer to the results published here, please use the following BibTeX entry. Sincerely appreciate it!
```shell
@article{zhang2023patch,
  title={Patch-based Selection and Refinement for Early Object Detection},
  author={Zhang, Tianyi and Kasichainula, Kishore and Zhuo, Yaoxin and Li, Baoxin and Seo, Jae-Sun and Cao, Yu},
  journal={arXiv preprint arXiv:2311.02274},
  year={2023}
}
```
