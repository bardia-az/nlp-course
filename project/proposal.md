# Stock Market Prediction using Textual Data

## Abbstract

### Motivation
In the field of the stock market, fundamental analysis is often neglected, especially by amateur traders, due to its difficulty, and people tend to rely more on technical analysis, wherein all you need is just the price chart of a stock. Conversely, fundamental analysis, which is a cornerstone of investment strategy, aiming to evaluate the intrinsic value of a stock based on a company's financial health, entails a comprehensive monitoring of the company's news, annual reports, corporate events, etc. Extracting meaningful insights from this type of information manually, which is entirely textual, is cumbersome. However, we can utilize NLP models to exploit these kinds of textual information and predict the trend of a stock. In an ideal case, such models can be trained on large-scale data (e.g. on the economic and geopolitical news on the web) and observe the trend of each stock both short-term and long-term. In this project, however, we aim to use a model that analyzes stock-specific news and predicts the temporary mispricing that may occur due to a corporate event on a specific stock.

### Approach
Our idea is to consider short-term stock prediction as a classification problem with three classes based on specific news about the stock. The classes determine whether the price would be increasing, decreasing, or neutral. We plan to use a pre-trained large language model like BERT or GPT-2, and continue pre-training it on a context related to the stock market in order to put the model in this space first. To tailor the model to our needs, ideas like fine-tuning the whole or part of the model, prompt engineering, prefix tuning, adaptor training, etc., can be adopted.

### Data
There is a nice dataset called [EDT](https://github.com/Zhihan1996/TradeTheEvent/tree/main/data), which contains three subsets. One can be used for *Domain Adaptation*, which includes financial news articles and a financial terms encyclopedia. The second set can be used for *Corporate Event Detection*, such as Acquisition, Dividend Cut, and so forth. The last set, *Trading Benchmark*, contains 303,893 news articles with timestamps, and a ticker tag that has been automatically assigned to the news with the corresponding price data, up to 3 days after the news is published. One can use an event detector trained on the second subset and use its output as side information for the task associated with the *Trading Benchmark*. However, we may ignore the second set and optimize the model on the third set in a way that it could infer such events implicitly.