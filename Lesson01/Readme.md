# notebook for lesson 01 

Dataset inspection checklist

1. What is the dataset for?
   - pretraining?
   - supervised fine-tuning?
   - preference training?
   - evaluation?
   - RAG corpus?

2. What is the license?
   - Can I use it commercially?
   - Are there attribution or usage limits?

3. What are the splits?
   - train
   - validation
   - test
   - sft
   - preference
   - ranking

4. What are the columns?
   - text
   - prompt
   - response
   - messages
   - chosen / rejected
   - language
   - source
   - score

5. What is the shape?
   - number of rows
   - size on disk
   - average text length
   - token length distribution

6. What is the quality story?
   - filtered?
   - deduplicated?
   - human-written?
   - synthetic?
   - multilingual?
   - safety-filtered?

7. What can go wrong?
   - duplicates
   - empty rows
   - very short rows
   - very long rows
   - train/test leakage
   - toxic or low-quality text
   - benchmark contamination
   - bad prompt/response formatting


   
