import helpers

import numpy as np
import tensorflow as tf
import helpers

train_onehot=42278
test_onehot=39565
EOS=39566

def read_data(source_file, target_file):
    feature_set = []
    label_set = []
    count = 0
    with open(source_file) as source_f:
        with open(target_file) as target_f:
            source, target = source_f.readline(), target_f.readline()
            while source and target:
                source_ids = map(int,source.split(' '))
                target_ids = map(int,target.split(' '))
                # target_ids.append(EOS)
                feature_set.append(source_ids)
                label_set.append(target_ids)
                count += 1
                source, target = source_f.readline(), target_f.readline()
                if count%100 == 0:
                    output_feat, out_label = feature_set, label_set
                    feature_set, label_set = [],[]
                    yield(output_feat, out_label)
                
    # return feature_set, labe



y = read_data('2p.train.onehot','train.seq2seq.txt.onehot')
train_feat, train_labels = next(y)

# print train_feat, train_labels

xt, x_len = helpers.batch(train_feat)
yt, y_len = helpers.batch(train_labels)
tf.reset_default_graph()
sess = tf.InteractiveSession()


PAD = 0
# EOS = 1

vocab_size = 42278
input_embedding_size = 300

encoder_hidden_units = 128
decoder_hidden_units = encoder_hidden_units


encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')


embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

# Sharing embeddings
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)


encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)

del encoder_outputs

# encoder_final_state

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,
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

# batch_ = [[6], [3, 4], [9, 8, 7]]
batch_ = train_feat
print(train_feat)
batch_, batch_length_ = helpers.batch(batch_)
print('batch_encoded:\n' + str(batch_))

din_, dlen_ = helpers.batch(train_labels)
print('decoder inputs:\n' + str(din_))

pred_ = sess.run(decoder_prediction,
    feed_dict={
        encoder_inputs: batch_,
        decoder_inputs: din_,
    })
print('decoder predictions:\n' + str(pred_))

def next_feed():
    encoder_inputs_, decoder_i = next(y)
    encoder_inputs_, _ = helpers.batch(encoder_inputs_)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] for sequence in decoder_i]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in decoder_i]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

loss_track = []
max_batches = 3000
batches_in_epoch = 100

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
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')

