import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer # Converter uma coleção de textos em uma matriz de contagem de tokens (palavras)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from googletrans import Translator

# Carregar os feedbacks de filmes
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories() 
             for fileid in movie_reviews.fileids(category)]
            
            # Cada item na lista documents é uma tupla
            # primeira parte da tupla é uma lista das 
            # palavras de uma crítica, e a segunda parte
            # é a categoria da crítica ("pos" ou "neg") 
            
# Junta as palavras e categorias
texts =[' '.join(doc) for doc, _ in documents] # Une as palavras de cada crítica em uma única string separada por espaços
labels = [category for _, category in documents]

vectorizer = CountVectorizer(ngram_range=(1, 2), max_df= 0.8,  min_df=1)
X = vectorizer.fit_transform(texts) # Matriz de caracteristicas
y = labels # Pos ou neg

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina o modelo
model = MultinomialNB(alpha=0.8)
model.fit(X_train, y_train)

# Previsoes
y_pred = model.predict(X_test)

print("Acurácia: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

translator = Translator()

def translate_comment(comment):
    translated = translator.translate(comment, src='auto', dest='en')
    return translated.text

def preprocess_text(text):
    # Exemplo de pré-processamento
    text = text.lower()  # Converter para minúsculas
    text = ' '.join(word for word in text.split() if word.isalpha())  # Remover pontuação e números
    return text

def predict_sentiment(comment):
    translated_comment = translate_comment(comment)
    processed_comment = preprocess_text(translated_comment)
    comment_vector = vectorizer.transform([processed_comment])
    prediction = model.predict(comment_vector)
    return prediction[0]

while(True):
    new_comment = input("Faça um teste com seu comentário (Digite '0' para sair): ")
    if(new_comment == "0"):
        print("Encerrado!")
        break
    print("Sentimento: ", predict_sentiment(new_comment))

