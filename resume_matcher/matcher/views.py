

# Create your views here.
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_url = fs.url(filename)

        # Perform similarity calculation
        resume_directory = settings.MEDIA_ROOT
        file1_path = os.path.join(resume_directory, filename)

        if not os.path.isfile(file1_path):
            return render(request, 'matcher/error.html', {'message': 'Resume file not found.'})

        directory_path = "matcher/jobs"
        max_similarity = 0.0
        most_similar_file = None
        all_files = {}

        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
            preprocessed_text = " ".join(tokens)
            return preprocessed_text

        def calculate_tfidf_similarity(file1_path, file2_path):
            with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
                file1_contents = file1.read()
                file2_contents = file2.read()
                
                file1_preprocessed = preprocess_text(file1_contents)
                file2_preprocessed = preprocess_text(file2_contents)
                
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([file1_preprocessed, file2_preprocessed])
                
                file1_tfidf = tfidf_matrix[0]
                file2_tfidf = tfidf_matrix[1]
                
                similarity = (file1_tfidf * file2_tfidf.T).toarray()[0, 0]
                
                return similarity

        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file2_path = os.path.join(directory_path, filename)
                
                similarity = calculate_tfidf_similarity(file1_path, file2_path)
                filename_without_extension = os.path.splitext(filename)[0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_file = filename_without_extension
                
                all_files[filename_without_extension] = similarity

        all_files_sorted = sorted(all_files.items(), key=lambda x: x[1], reverse=True)

        data_top = [["Job", "Similarity"]]
        data_bottom = [["Job", "Similarity"]]

        for file_name, similarity in all_files_sorted[:5]:
            data_top.append([file_name, similarity])

        for file_name, similarity in all_files_sorted[-5:][::-1]:
            data_bottom.append([file_name, similarity])

        return render(request, 'matcher/result.html', {
            'most_similar_file': most_similar_file,
            'max_similarity': max_similarity,
            'data_top': data_top,
            'data_bottom': data_bottom,
            'file_url': file_url
        })
    return render(request, 'matcher/upload.html')
