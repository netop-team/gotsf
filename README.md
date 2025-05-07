<div align="center">
<h1>Goal-Oriented Time-Series Forecasting: Foundation Framework Design</h1>

[![paper](https://img.shields.io/static/v1?label=arXiv&message=2402.03885&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2504.17493)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/papers/2504.17493)
[![5G Dataset](https://img.shields.io/badge/Dataset-Hugging_Face-blue)](https://huggingface.co/datasets/netop/Beam-Level-Traffic-Timeseries-Dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/MIT)

</div>
This repository contains the official implementation of the paper:

   >L. Fechete*†, M. Sana†, F. Ayed†, N. Piovesan†, W. Li†, A. De Domenico†, T. Si Salem†‡.
   [Goal-Oriented Time-Series Forecasting: Foundation Framework Design](https://arxiv.org/pdf/2504.17493).
   ><br>*École Polytechnique, Palaiseau, France (Research Intern), †Paris Research Center, Huawei Technologies, Boulogne-Billancourt, France, ‡Lead researcher for this study.

### 📝 Abstract:
Traditional time-series forecasting often focuses only on minimizing prediction errors, ignoring the specific requirements of real-world applications that employ them. This paper presents a new training methodology, which allows a forecasting model to dynamically adjust its focus based on the importance of forecast ranges specified by the end application. Unlike previous methods that fix these ranges beforehand, our training approach breaks down predictions over the entire signal range into smaller segments, which are then dynamically weighted and combined to produce accurate forecasts. We tested our method on standard datasets, including a new dataset from wireless communication, and found that not only it improves prediction accuracy but also improves the performance of end application employing the forecasting model. This research provides a basis for creating forecasting systems that better connect prediction and decision-making in various practical applications.

![main figure](figures/system_model.png)
