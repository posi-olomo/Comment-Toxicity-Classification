# Comment-Toxicity-Classification

About 37% of young people between the ages of 12 and 17 have been bullied online. 30% have had it happen more than once.

<p align="center">
  <img src="https://github.com/posi-olomo/Comment-Toxicity-Classification/assets/75603128/212f1cac-8244-4fed-a318-194419c9bf12" style="height: 500px; width:500px"/>
</p>

<p align="center">
  Photo by <a href="https://unsplash.com/@a_d_s_w?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Adrian Swancar</a> on <a href="https://unsplash.com/photos/JXXdS4gbCTI?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
</p>

Cyber Bullying is a priority for every social media company at the moment from Google to Twitter. Extending past Social Media companies, every organization with a website that allows users to comment takes cyberbullying very seriously, from schools to company websites. It is important that the safety of workers and users of the website are taken seriously and protected.

Cyberbullying occurs on every platform and in every single country in the world. As a company, it is your duty to ensure that nasty comments are flagged and taken off the platform. To be able to do that you need a deep learning algorithm that can detect when a comment is toxic and its class(es) of toxicity.

That is exactly what my web app does: you upload a comment and it tells you if it is clean or if it is toxic and its class(es) of toxicity [toxic, severe_toxic, obscene, threat, insult, and identity_hate].

Check it out here: 
https://comment-toxicity-classifier.onrender.com/

## Project Process

#### Data Cleaning &rarr; Text Vectorization &rarr; Model Architecture &rarr; Model Performance &rarr; Deployment

The data was sourced from Kaggle: 
https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge
___
### Data Cleaning
I cleaned the data by removing hyperlinks, special characters and numbers. 
___
### Text Vectorization
I used the Tf-Idf Text Vectorizer, which helps us to vectorize our input data into a specific number of tokens.
___
### Model Architecture
I built a 3-layer neural network, with 2 hidden layers with the ReLU activation function. The output layer had a Sigmoid activation function. 

To ensure that the model could multi-classify, I compiled the model with a binary_crossentropy loss.
___
### Model Performance
The model performed well with a 
* Training Accuracy: 99.9%
* Testing Accuracy: 93.2%


### Deployment
The model was deployed via a Flask app and hosted using Render.

### References
https://www.slicktext.com/blog/2020/05/cyberbullying-statistics-facts/#:~:text=About%2037%25%20of%20teens%20between%20the%20ages%20of,and%20perpetrators%20of%20cyberbullying%20in%202019%20and%202020.
