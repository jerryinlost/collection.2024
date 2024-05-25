from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# def extract_features(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split

# # Load the dataset
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(interactions[['user_id', 'item_id', 'rating']], reader)

# # Split the dataset
# trainset, testset = train_test_split(data, test_size=0.25)

# # Train the model
# algo = SVD()
# algo.fit(trainset)

# # Make predictions
# predictions = algo.test(testset)

# # Using implicit feedback with Implicit:

# import implicit
# from scipy.sparse import coo_matrix

# # Prepare the data
# user_item_matrix = coo_matrix((interactions['rating'], (interactions['user_id'], interactions['item_id'])))

# # Train the model
# model = implicit.als.AlternatingLeastSquares(factors=50)
# model.fit(user_item_matrix)

# # Recommend items for a user
# user_id = 1
# recommendations = model.recommend(user_id, user_item_matrix)