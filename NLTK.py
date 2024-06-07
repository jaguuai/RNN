import nltk
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sentence="""
Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines,
particularly computer systems. It is a field of research in computer science that develops 
and studies methods and software that enable machines to perceive their environment 
and uses learning and intelligence to take actions that maximize their chances of achieving defined goals.[1] 
Such machines may be called AIs.
AI technology is widely used throughout industry, government, and science.
Some high-profile applications include advanced web search engines (e.g., Google Search); 
recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, 
Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); 
and superhuman play and analysis in strategy games (e.g., chess and Go).[2] However,
 many AI applications are not perceived as AI: "A lot of cutting edge AI has filtered into general applications, 
 often without being called AI because once something becomes useful enough and common enough
 it's not labeled AI anymore."[3][4]
 """
tokens=nltk.word_tokenize(sentence)
# print(tokens)
tagged=nltk.pos_tag(tokens)
# print(tagged)
entities=nltk.chunk.ne_chunk(tagged)
print(entities)