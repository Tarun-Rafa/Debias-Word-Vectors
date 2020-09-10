# Debiasing word vectors
<p><a href="https://en.wikipedia.org/wiki/GloVe_(machine_learning)">GloVe word embeddings</a> are explored for gender bias and subsequently a debiasing operation is performed on the word vectors to eliminate the bias. Since word embeddings are very computationally expensive to train, we will load a 50-dimensional <a href="https://nlp.stanford.edu/projects/glove/">pre-trained GloVe vectors</a> to represent word embeddings. This project is implemented as part of the <a href="https://github.com/adityachandupatla/deeplearning_coursera">deeplearning specialization</a> from Coursera.</p>
<h2>Running the Project</h2>
<ul>
  <li>Make sure you have Python3 installed</li>
  <li>Clone the project to your local machine and open it in one of your favorite IDE's which supports Python code</li>
  <li>Make sure you have <a href="http://www.numpy.org/">Numpy</a> dependency installed</li>
  <li>Run debias_word_vectors.py</li>
</ul>
If you find any problem deploying the project in your machine, please do let me know.

<h2>Technical Skills</h2>
This project is developed to showcase my following programming abilities:
<ul>
  <li>Python</li>
  <li>Working with word embeddings</li>
</ul>

<h2>Development</h2>
<ul>
  <li>Sublimt Text has been used to program the application. No IDE has been used.</li>
  <li>Command line has been used to interact with the application.</li>
  <li>The project has been tested on Python3 version: 3.6.1.</li>
</ul>

<h2>Working</h2>
<p><b>Dataset characteristics</b>: Before you can run the application locally, ensure that you have downloaded the dataset from the above specified link. Within the website search for pre-trained word vectors and download glove.6B.zip. Unzip it and place the file 'glove.6B.50d.txt' in the 'data' folder within this project. The content of the dataset is as follows:
  <ul>
    <li>words: set of words in the vocabulary. Total count: 400,000 words</li>
    <li>word_to_vec_map: dictionary mapping words to their GloVe vector representation</li>
  </ul>
</p>

<p>We will be using <a href="https://en.wikipedia.org/wiki/Cosine_similarity">Cosine similarity</a> to measure how similar two words are.</p>

<p>We will first examine gender biases that can be reflected in a word embedding. We will first compute a vector g = e_woman − e_man , where  e_woman  represents the word vector corresponding to the word woman, and  e_man corresponds to the word vector corresponding to the word man. The resulting vector g roughly encodes the concept of "gender". After running the application you will see, the astonishing results, which show us how biased these word vectors are in reflecting unhealthy gender stereotypes. For example, "computer" is closer to "man" while "literature" is closer to "woman".</p><br/>

<p><b>Neutralize bias for non-gender specific words</b>: We'll use an algorithm by <a href="https://arxiv.org/abs/1607.06520">Boliukbasi et al., 2016</a> to perform gender debiasing. Note that some word pairs such as "actor"/"actress" or "grandmother"/"grandfather" should remain gender specific, while other words such as "receptionist" or "technology" should be neutralized, i.e. not be gender-related.</p><br/>

<p><b>Equalization algorithm for gender-specific words</b>: Equalization is applied to pairs of words that you might want to have differ only through the gender property. As a concrete example, suppose that "actress" is closer to "babysit" than "actor." By applying neutralizing to "babysit" we can reduce the gender-stereotype associated with babysitting. But this still does not guarantee that "actor" and "actress" are equidistant from "babysit." The equalization algorithm takes care of this. The key idea behind equalization is to make sure that a particular pair of words are equidistant from the gender concept encoded by the word embeddings.</p>

<h2>TODO</h2>
<ul>
  <li>These debiasing algorithms are very helpful for reducing bias, but are not perfect and do not eliminate all traces of bias. One weakness of this implementation was that the bias direction 'g' was defined using only the pair of words {"woman" and "man"}. If 'g' were defined by computing g1 = e_woman − e_man ;  g2 = e_mother − e_father ;  g3 = e_girl − e_boy ; and so on and averaging over them, we would have obtained a better estimate of the "gender" dimension.</li>
</ul><br/>
Use this, report bugs, raise issues and Have fun. Do whatever you want! I would love to hear your feedback :)

~ Happy Coding
