# Probabilistic Forecasting of Renewable Energy Generation and Wholesale Market Prices Using Quantile Regression in Keras
:rocket: Blog post on personal website :link: [Probabilistic Forecasting of Renewable Generation & Wholesale Prices with Quantile-Regression](https://richardfindlay.co.uk/probabilistic-forecasting-of-renewable-generation-and-wholesale-prices-with-quantile-regression-2)

<p align="center">
  <img src="https://github.com/RichardFindlay/day-ahead-probablistic-forecasting-with-quantile-regression/blob/main/visualisations/d3_quantile_plot_examples.png" />
  screenshot of interactive d3.js plots illustrating probabilistic forecasting performance
</p>

### Project Description :open_book::
This repository demonstrates the use of deep learning techniques in combination with quantile regression to produce probabilistic forecasts. The above figure depicts the consecutive DA quantile forecasts for each of the investigated variables over one week, with further quantification and discussion given on the forecast performance given in the the accompanying [blog post](https://richardfindlay.co.uk/probabilistic-forecasting-of-renewable-generation-and-wholesale-prices-with-quantile-regression-2).

The code investigates the performance of four different deep-learning architectures; Bi-directional LSTM, Seq-2-Seq, Seq-2-Seq with Temporal Attention and Seq-2-Seq with Temporal and Spatial Attention. To help give context, comparisons are made to a simplistic daily persistence forecasting technique, as well as to the Transmission System Operator's forecast (TSO). The models are predicated off the notion that there is an increased complexity added at each iteration, which accompanied the hypothesis that an increased performance should be observed between each iteration, which was not the case when test performance was investigated.

<p align="center">
  <img src="https://github.com/RichardFindlay/day-ahead-probablistic-forecasting-with-quantile-regression/blob/main/visualisations/model_architecture_schematic_markup.png" />
  model architecture schematic for encoder-decoder with spatial and temporal attention mechanisms as implemented in keras
</p>

### Performance Overview :racing_car::
The above figure illustrates the pinicale of the model complexity investigated as part of this project. With both temporal and spatial attention mechanisms, the novel encoder-decoder architecture does not always prevail as the best preforming technique but shows encourging performance and may merit further investigation and and fine-tuning. 

<p align="center">
  <img src="https://github.com/RichardFindlay/day-ahead-probablistic-forecasting-with-quantile-regression/blob/main/visualisations/d3_temporal_attention_plot_solar.png" />
</p>


<p align="center">
  <img src="https://github.com/RichardFindlay/day-ahead-probablistic-forecasting-with-quantile-regression/blob/main/visualisations/solar_spatial_attentions_animation.gif" width="450"/>
</p>



### Resources :gem:: 
+ [https://www.elexon.co.uk/documents/training-guidance/bsc-guidance-notes/bmrs-api-and-data-push-user-guide-2/](https://www.elexon.co.uk/documents/training-guidance/bsc-guidance-notes/bmrs-api-and-data-push-user-guide-2/)
+ [https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
+ [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview)
+ [https://colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs)
+ [https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
+ [https://colab.research.google.com/github/kmkarakaya/ML_tutorials/blob/master/seq2seq_Part_D_Encoder_Decoder_with_Teacher_Forcing.ipynb](https://colab.research.google.com/github/kmkarakaya/ML_tutorials/blob/master/seq2seq_Part_D_Encoder_Decoder_with_Teacher_Forcing.ipynb)
