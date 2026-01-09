# Dense Query Representations Alternative to Text Queries for Dense Conversational Search

## Environments
I recommend to install requirements with:
```
pip install requirements.txt
```

## Index
run scripts:
```
bash scripts/encode.sh $INDEX_DIR $MODEL_NAME_OR_PATH
```

## Training
run scripts:
```
bash scripts/train.sh $OUTPUT_DIR $MODEL_NAME_OR_PATH
```

## Search
run scripts:
```
bash scripts/search.sh $SEARCH_FILE $INDEX_DIR $MODEL_NAME_OR_PATH
```