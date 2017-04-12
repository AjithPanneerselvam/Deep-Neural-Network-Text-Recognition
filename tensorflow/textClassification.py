import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups


# Building neural network

categories = ["comp.graphics","sci.space","rec.sport.baseball"]

newsgroups_train = fetch_20newsgroups(subset = 'train',
                                     categories = categories)
newsgroups_test = fetch_20newsgroups(subset = 'test', categories = categories)


vocab = Counter()

for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)

def get_word_2_index(vocab):
    word2index = {}

    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index

word2index = get_word_2_index(vocab)


def get_batch(df,i,batch_size):
    batches = []
    results = []
    texts = df.data[i*batch_size : i*batch_size + batch_size]
    categories = df.target[i*batch_size : i*batch_size + batch_size]
    
    for text in texts:
        layer = np.zeros(total_words, dtype = float)

        for word in text.split(' '):
            layer[word2index[word.layer()]] += 1

        batches.append(layer)
        
    for category in categories:
        y = np.zeros((3),dtype = float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)
        
    return np.array(batches), np.array(results)

    
# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# Neural Network parameters
n_hidden_1 = 10
n_hidden_2 = 5
n_input = total_words
n_classes = 3

input_tensor = tf.placeholder(tf.float32, [None, n_input], name = "input")
output_tensor = tf.placeholder(tf.float32, [None, n_classes], name = "output")


def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_addition)
    
    # Hidden layer with RELu (Rectified Linear unit) activation function
    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b1'])
    layer_2_activation = tf.nn.relu(layer_2_addition)
    
    # Output layer with linear activation
    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']
    
    return out_layer_addition

    
# Defining tensor variables
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
prediction = multilayer_perceptron(input_tensor,weights,biases)

# Define loss (Softmass activation function for the output layer)
entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits = prediction,
                                                       labels = output_tensor)
loss = tf.reduce_mean(entropy_loss)

# Adam Moment Estimation method to find gradient descent and apply it
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_intializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(newsgroups_train) / batch_size)
        # Loop over all batches
        
        for i in range(total_batch):
            batch_x, batch_y = get_batch(newsgroups_train, i, batch_size)
            # Run optimization op (backdrop) and cost op (to get loss value)
            c,_ = session.run([loss,optimizer], feed_dict = {input_tensor: batch_x, output_tensor: batch_y})
            # Computer average loss
            avg_cost += c / total_batch
            
        # Display logs per epoch step
        #if epoch % display_step == 0:
         #   print(("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_cost))
        
       # print ("Optimization Finished!")
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Calculate accuracy 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test, batch_y_test = get_batch(newsgroups_test, 0, total_test_data)
    print ("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))