# Network Security Project S22

### Using Machine Learning to identify Botnet DNS Activity

- front-end.py
  
  - Main application for parsing PCAP files for bot activity

- cnn_chrlevel_classifier.py
  
  - Generates character-level CNN model 

- lstm_classifier.py
  
  - Generates character-level LSTM model using word embeddings

- IDN_find.py
  
  - Uses google search engine to find IDN/punycode domains

- IDN_parse.py

  - Formats the IDN domains for consistency 

- IDN_test.py

  - Tests both character-level encoded models on the IDN dataset

- datasets/
  
  - Directory containing datasets for model training. All files are saved in CSV format

- demo/
  
  - Directory containing files related to DNS server demonstration in course Presentation

