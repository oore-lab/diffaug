# DiffAug: A Diffuse-and-Denoise Augmentation for Training Robust Classifiers [NeurIPS 2024]

[OpenReview](https://openreview.net/forum?id=Tpx9gcZVBf) [Poster](https://github.com/chandramouli-sastry/diffaug_website/blob/main/static/pdfs/poster_final.pdf)

Code coming soon.

---
## <p align='center'>Key Idea: Use <b><i>intermediate-images/partially-synthesized</i></b> diffusion-samples to train robust classifiers.</p>

<p align='center'><img src='https://github.com/chandramouli-sastry/diffaug_website/blob/main/static/images/diffaug_illustration.jpg' width='50%'/></p>

---

<b>Highlights:</b>
1. Simple and compute-efficient augmentation to enhance robustness to <i><b>covariate-shifts, OOD-examples & adversarial-examples</b></i>.
2. Achieves improvements with _**no extra training data**_ -- i.e., Diffusion-model has access to the same dataset as the classifier.
4. Can also be used for **_test-time adaptation_**!
5. Remains effective even when high-quality synthetic data from larger diffusion-model is available!
6. We show Theoretical and Empirical connections with ***Perceptually Aligned Gradients***. This can enhance Classifier-Guided Diffusion.

---

## <p align='center'>Results</p>

<p align='center'><img src="https://github.com/chandramouli-sastry/diffaug_website/blob/main/static/images/imnet_c2.png"></p>
<p align='center'><img src='https://github.com/chandramouli-sastry/diffaug_website/blob/main/static/images/glimpse_results.png'></p>
<p align='center' width='90%'><img src='https://github.com/chandramouli-sastry/diffaug_website/blob/main/static/images/pag_cg.png' />
DiffAug can be used to promote Perceptually Aligned Gradients and this can enhance classifier-guided generation. See Theorem 1 in paper for theoretical intuitions. 
</p>

---
