# NLP_begin

## Regular Expression
``` bash
ll : 폴더 내의 파일
# sys.argv[1] : refine.regex.txt
# sys.argv[2] : 1
python refine.py refine.regex.txt 1 < review.sorted.uniq.tsv > review.sorted.uniq.refiend.tsv
```

## Tokenization
### why? we need tokenization
- 두개 이상의 다른 token들의 결합으로 이루어진 단어를 쪼개어, vocabulary 숫자를 줄이고, 희소성(sparseness)을 낮추기 위함 

- 한국어는 교착어기 때문에 tokenization이 필요, 띄어쓰기 통일의 필요성

### 형태소 분석(mecab : 속도가 가장 빠름)
- 형태소(동사 / 명사)를 비롯하여, 어근, 접두사/접미사, 품사(POS, part-of-speech)등 다양한 언어적 속성의 구조를 파악하는 것

### 품사 태깅
- 형태소의 뜻과 문맥을 고려하여 그것에 마크업을 하는 일
```python3
# Sentence Segmentation
from nltk.tokenize import sent_tokenize
```
```bash
# mecab -O(option) wakati(one of options)
echo '아버지가 방에 들어가신다.' | mecab -O wakati
# cut -f2(두번째 컬럼)
cut -f2 ./review.sorted.uniq.refined.tsv
# paste file1 file2 파일 두개를 덧붙임
paste file1 file2
```

## Subword Segmentation
- subword : 단어보다 더 작은 의미 단위
- Oov(Out Of Vocabulary) : training과정에 없던 데이터
### 한국어의 경우
- 띄어쓰기가 제멋대로인 경우가 많으므로, normalization 없이 바로 subword segmentation을 적용하는 것은 위험

- 따라서 형태소 분석기를 통한 tokenization을 진행한 이후, subword segmentation을 적용하는 것을 권장

## Detokenization
