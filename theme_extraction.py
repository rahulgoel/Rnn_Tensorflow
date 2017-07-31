import gzip
import nltk
import string
import json
import gensim
import random
import tensorflow as tf
from  helpers import *
import numpy
from nltk.tokenize import RegexpTokenizer
import itertools

auto = parse('reviews_Automotive_5.json.gz')    
beauty = parse('reviews_Beauty_5.json.gz')
electronics = parse('reviews_Electronics_5.json.gz')
home = parse('reviews_Home_and_Kitchen_5.json.gz')
pet = parse('reviews_Pet_Supplies_5.json.gz')    
model_word2vec = gensim.models.Word2Vec.load_word2vec_format('glove.6B.100d.txt', binary=False)  
tokenizer = RegexpTokenizer(r'\w+')


text_review = []
text_summary = []
vocab_set = set()
# Corpus size, effectively toy data
corpus_size = 30000
for i in range(corpus_size):
    b = random.choice([auto, home, electronics, pet, beauty])
    a = b.next() 
    text = a['reviewText']
    text_target = a['summary']
    text_review.append(text)
    text_summary.append(text_target)
    vocab_set.update(tokenizer.tokenize(text))

# Dictionaries to track words
vocab_dict = {j:i for i,j in enumerate(vocab_set)}
rev_vocab_dict ={i:j for i,j in enumerate(vocab_set)}
vocab_size = len(vocab_set)
input_embedding_size = 100
encoder_hidden_units = 128
decoder_hidden_units = encoder_hidden_units
EOS = vocab_size
PAD = vocab_size-1
loss_track = []
max_batches = 300000
batches_in_epoch = 100



print("example candidate words", extract_candidate_words(random.choice(text_review)))
print("example candidate phrases", extract_noun(random.choice(text_review)))


corpus, dic, model = score_keyphrases_by_tfidf(text_review, candidates = 'words')
print("Extraction Example 1, LDA on salient terms")
print(model.print_topics(-1))
top_words = [[w for w, word in model.show_topic(topicno, topn=5)] for topicno in range(model.num_topics)]
for top in top_words:
    print(top)
print("Extraction Example 2: LDA on noun phrases")
corpus, dic, model = score_keyphrases_by_tfidf(text_review, candidates = 'chunks')
print(model.print_topics(-1))
top_words = [[w for w, word in model.show_topic(topicno, topn=5)] for topicno in range(model.num_topics)]
for top in top_words:
    print(top)
# Uncomment to print words with scores5B
# for i in corpus:
#     j = sorted(i, reverse = True, key = lambda x:x[1])
#     for word, score in j:
#         print dic[word], score
#         print "\n"
############################################################################        
# PART2
#Seq2Seq Model initialized with word embeddings to summarize on summaries taking topic information into account
def batch_data(text, text_target):
    # use iterators later
    batch_text, batch_target, topic_list = [], [], []
    for review, summary in random.sample(zip(text, text_target),5):
        out, su,top = [], [], []
        #Get topic here as well
        tokenized = tokenizer.tokenize(review)
        bow_vector = dic.doc2bow(tokenized)
        lda_vector = model[bow_vector]
        topic, prob = model.show_topic(max(lda_vector, key=lambda item: item[1])[0])[0]
        if topic in vocab_dict:
            t1 = vocab_dict[topic]
        else:
            t1 = vocab_dict['unk']
        
        for token in tokenized:
            if token in vocab_dict:
                out.append(vocab_dict[token])                
        for t in tokenizer.tokenize(summary):
            if t in vocab_dict:
                su.append(vocab_dict[t])
                top.append(t1)
        batch_text.append(out)
        batch_target.append(su)
        topic_list.append(top)
    return batch_text, batch_target, topic_list

train_feat, train_labels, topic_list = batch_data(text_review, text_summary)


tf.reset_default_graph()
sess = tf.InteractiveSession()

embedding = numpy.zeros((vocab_size+1, input_embedding_size))
for word in vocab_dict:
    index = vocab_dict[word]
    if word in model_word2vec:
        embedding[index] = model_word2vec[word]
    else:
        embedding[index] = model_word2vec['unk']
del model_word2vec
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
encoder_topic = tf.placeholder(shape=(None , None), dtype=tf.int32, name='encoder_topic')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')


# Initialize vocab to glove
embeddings = tf.get_variable(initializer=  tf.constant_initializer(embedding), name="W", shape=embedding.shape, trainable=False)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
encoder_topic_embedded = tf.nn.embedding_lookup(embeddings, encoder_topic)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)
# We dont need encoder outputs
del encoder_outputs

# Concat topic and decoder inputs 
final_decoder_input = tf.concat([decoder_inputs_embedded, encoder_topic_embedded],2)
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, final_decoder_input,
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder",
)

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)

# decoder_logits
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.global_variables_initializer())

batch_ = train_feat
batch_, batch_length_ = batch(batch_)
print('batch_encoded:\n' + str(batch_))
din_, dlen_ = batch(train_labels)
print('decoder inputs:\n' + str(din_))
tin_, tlen_ = batch(topic_list)
print('decoder inputs:\n' + str(tin_))
pred_ = sess.run(decoder_prediction,
    feed_dict={
        encoder_inputs: batch_,
        decoder_inputs: din_,
        encoder_topic: tin_
    })
print('decoder predictions:\n' + str(pred_))


def next_feed():
    encoder_inputs_, decoder_i, enc_t_ = batch_data(text_review, text_summary)
    encoder_inputs_, _ = batch(encoder_inputs_)
    enc_topics_, _ = batch(
        [[EOS]+ (sequence) for sequence in enc_t_])
    decoder_targets_, _ = batch(
        [(sequence) + [EOS] for sequence in decoder_i]
    )
    decoder_inputs_, _ = batch(
        [[EOS] + (sequence) for sequence in decoder_i]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
        encoder_topic: enc_topics_
    }


try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                input_text = [rev_vocab_dict[i] for i in inp if i != 0]
                predicted_text = [rev_vocab_dict[i] for i in pred if i != 0]
                print(input_text, predicted_text)
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')






















    
