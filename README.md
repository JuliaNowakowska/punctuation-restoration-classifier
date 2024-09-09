# Punctuation restoration classifier
Large language models can be effectively used in classification tasks, either directly - in few-shot setups, or indirectly - by generating pseudo labels that are then used to train custom models. An example of architecture that utilizes the latter approach is a pre-trained language model combined with a single linear, classification layer on top.
It can be used for punctuation restoration task as a model that predicts punctuation mark class to be inserted after each word of the input sentence.

## Training data
The data used for training the model comes from PolEval 2021 challenge: https://github.com/poleval/2021-punctuation-restoration I have set it up so that the target labels correspond to the punctuation marks that should follow each word in the input sentence.
| Input   | Target labels  |
|------------|------------|
| w wywiadzie dla polski jarosław kaczyński podkreślił że informacje dotyczące radosława sikorskiego zagrażają interesowi państwa to naprawdę wszystko co mogę na ten temat powiedzieć odpowiedział gdy dziennikarz pytał o bardziej szczegółowe informacje premier kaczyński sugeruje że dobry kandydat po na szefa dyplomacji to np jacek saryusz wolski wymieniony polityk zyskał uznanie braci kaczyńskich za dotychczasową działalność w charakterze dyplomaty i dużą wiedzę premier krytycznie ocenia również dotychczasowe poczynania po gabinet cieni i inne podobne pomysły okazały się fikcją świadczą o tym kłopoty ze znajdywaniem kolejnych ministerstw cały czas nie wiadomo kto obsadzi które ministerstwo a tam gdzie już wiadomo to nie ma się z czego cieszyć mówił kaczyński stwierdził także że uważny obserwator życia publicznego musi dostrzegać też rolę w tym wszystkim jana krzysztofa bieleckiego będzie osobą bardzo ważną pytany o radę jaką dałby donaldowi tuskowi powiedział radzę donaldowi tuskowi żeby uczciwie przyjrzał się temu co się wydarzyło w polsce przez ostatnie dwa lata żeby jednak zrzucił czarne okulary choć wcale nie musi zakładać różowych i rozejrzał się dookoła i żeby w końcu doszedł do wniosku że wiele rzeczy warto kontynuować|O O O O O O A O O O O O O O D O O A O O O O O H A O O O O O O D O O A O O O O O O O O D O H O O O O O O O O O O O O O O O D O O O O O O D O O O O O O O O D O O O O O O O D O O O A O O O D O A O O A O O O O O O H O D O A O O O O O O O O O O O O O O D O O O D O O A O O O A N O O A O O O O A O O O O O O O O D O O O O A O O O O O A O O O D O O O O O O A O O O O D|

## Evaluation on Wikipunct benchmark
Following table shows the results achieved by various pre-trained LLMs trained for 30 epochs using the architecture and data described above.
| Model               | Hyphen | Comma | Full stop | Question mark | Colon |
|---------------------|--------|-------|----------|-----------|---------------|
| baseline            | 66.91  | 76.08 | 88.86     | 80.61         | 82.98 |
| multilingual BERT   | 35.97  | 63.4  |68.64     | 37.89         | 58.33 |
| HerBERT base        | 48.0   | 66.63 |81.35     | 57.57         | 57.46 |
| HerBERT large       | 49.25  | 70.41 |82.57     | 70.5          | 72.65 |
| XLM-RoBERTa base    | 37.22  | 67.47 |61.54     | 49.39         | 58.55 |
| XLM-RoBERTa large   | 44.84  | 71.47 |65.43     | 63.46         | 65.84 |

As a baseline model, I referred to the solution outlined in the paper: http://poleval.pl/files/poleval2021.pdf#page=33
