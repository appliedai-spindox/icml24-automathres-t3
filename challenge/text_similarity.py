import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextSimilarity:

    def __init__(self, train_data: list[dict], test_data: list[dict]) -> None:

        self.train_data = train_data
        all_questions = [entry['question'] for entry in train_data] + [entry['question'] for entry in test_data]

        self.vectorizer = TfidfVectorizer()
        self.vectorized_questions = self.vectorizer.fit_transform(all_questions)


    def get_most_similar_problem(self, test_problem: dict) -> (dict, float):

        test_question_vector = self.vectorizer.transform([test_problem['question']])
        similarities = cosine_similarity(test_question_vector, self.vectorized_questions[:len(self.train_data)])[0]
        most_similar_index = np.argmax(similarities)
        most_similar_problem = self.train_data[most_similar_index]
        return most_similar_problem, similarities[most_similar_index]
