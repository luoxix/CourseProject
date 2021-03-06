import metapy
import os

def tokens_lowercase(doc):
	#Write a token stream that tokenizes with ICUTokenizer, 
	#lowercases, removes words with less than 2 and more than 5  characters
	#performs stemming and creates trigrams (name the final call to ana.analyze as "trigrams")
	'''Place your code here'''
	tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
	tok = metapy.analyzers.LowercaseFilter(tok)
	tok = metapy.analyzers.LengthFilter(tok, min=3, max=10000)
	tok = metapy.analyzers.Porter2Filter(tok)
	ana = metapy.analyzers.NGramWordAnalyzer(1, tok)
	unigrams = ana.analyze(doc)
	#leave the rest of the code as is
	tok.set_content(doc.content())
	tokens, counts = [], []
	for token, count in unigrams.items():
		for i in range(count):
			tokens.append(token)
	return tokens


file_dir = "./data/Afghanistan_dataset"
fout = open("./data/war_dataset.txt", "w")
file_list = os.listdir(file_dir)

doc = metapy.index.Document()
for file in file_list:
	fread = open(os.path.join(file_dir, file), "r", encoding='windows-1252').readlines()
	content = '0 '
	for line in fread:
		content += line.strip() + ' '
	
	fout.write(content +'\n')

file_dir = "./data/Iraq_dataset"
file_list = os.listdir(file_dir)

doc = metapy.index.Document()
for file in file_list:
	fread = open(os.path.join(file_dir, file), "r", encoding='windows-1252').readlines()
	content = '1 '
	for line in fread:
		content += line.strip() + ' '
	
	fout.write(content +'\n')
