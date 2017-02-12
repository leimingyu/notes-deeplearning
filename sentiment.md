## term
The process of computationally identifying and categorizing opinions expressed in a piece of text, 
especially in order to determine whether the writer's attitude towards a particular topic, product, etc., 
is positive, negative, or neutral.

## challenges
The input text of the word requency may contain noise, where non-meaningful words, such as empty space, comman, the, etc.,
have more weight/frequency than the meaningful words, such as nice, great, disappointed, etc.

Instead of using the frequency, use the word ( exist: 1, not-exist: 0) as the input of the neural nets. Let the NN figure out the weights of each word in your document.

## speedup
* avoid adding 0 to the weights  ( hint: find the index of 1)
* avoid multiplying 1 to the input data

## nn vs lexicon
Sarcasm can be difficult for lexicon to interpret. Neural nets can understand the subtlety.

